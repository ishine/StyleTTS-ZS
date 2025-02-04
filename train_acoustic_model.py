import torch
import time
import random
import torch.nn as nn
import os
import os.path as osp
from Dataset.meldataset import build_dataloader
from Utility.optimizers import build_optimizer
from Utility.utils import *
from Utility.load_models import *
from Utility.losses import *
import click
import yaml
import shutil
from monotonic_align import mask_from_lens
from pyannote.audio import Model as SVModel

from accelerate import Accelerator
from accelerate.utils import LoggerType
from accelerate import DistributedDataParallelKwargs

from torch.utils.tensorboard import SummaryWriter

import logging
from collections import deque
from accelerate.logging import get_logger
logger = get_logger(__name__, log_level="DEBUG")

@click.command()
@click.option('-p', '--config_path', default='Configs/config_acoustic.yml', type=str)
def main(config_path):
    config = yaml.safe_load(open(config_path))

    log_dir = config['log_dir']
    if not osp.exists(log_dir): os.makedirs(log_dir, exist_ok=True)
    shutil.copy(config_path, osp.join(log_dir, osp.basename(config_path)))
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(project_dir=log_dir, split_batches=True, kwargs_handlers=[ddp_kwargs])    
    if accelerator.is_main_process:
        writer = SummaryWriter(log_dir + "/tensorboard")

    # write logs
    file_handler = logging.FileHandler(osp.join(log_dir, 'train_acoustic.log'))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s', '%Y-%m-%d %H:%M:%S'))
    logger.logger.addHandler(file_handler)

    batch_size = config.get('batch_size', 2)
    device = accelerator.device                 #'cuda'
    epochs = config.get('epochs_tts', 100)
    log_interval = config.get('log_interval', 10) # log every 10 iterations
    saving_epoch = config.get('save_freq', 1) # save every 5 epochs
    
    data_params = config.get('data_params', None)
    train_path = data_params['train_data']
    val_path = data_params['val_data']
    
    train_list, val_list = get_data_path_list(train_path, val_path)

    train_dataloader = build_dataloader(train_list,
                                        batch_size=batch_size,
                                        num_workers=2,
                                        dataset_config={},
                                        device=device)

    val_dataloader = build_dataloader(val_list,
                                    batch_size=batch_size,
                                    validation=True,
                                    num_workers=0,
                                    device=device,
                                    dataset_config={})
    
    train_dataloader = accelerator.prepare(train_dataloader)
    val_dataloader = accelerator.prepare(val_dataloader)
    
    with accelerator.main_process_first():
         # load pretrained ASR model
        ASR_config = config.get('ASR_config', False)
        ASR_path = config.get('ASR_path', False)
        asr_model = load_ASR_models(ASR_path, ASR_config)
        #asr_model = accelerator.prepare(asr_model)

        # load pretrained F0 model
        F0_path = config.get('F0_path', False)
        F0_model = load_F0_models(F0_path)
        #F0_model = accelerator.prepare(F0_model)

    # i, (waves, texts, input_lengths, mels, mel_input_length, labels, ref_mels, ref_labels, adj_mels, adj_mels_lengths) = next(enumerate(train_dataloader))

    model = build_acoustic_model(asr_model, F0_model, training=True)
    
    for k in model:
        model[k] = accelerator.prepare(model[k])
    
    _ = [model[key].to(device) for key in model]
    
    scheduler_params = {
        "max_lr": float(config['optimizer_params'].get('lr', 1e-4)),
        "pct_start": float(config['optimizer_params'].get('pct_start', 0.0)),
        "epochs": epochs,
        "steps_per_epoch": len(train_dataloader),
    }

    scheduler_params_dict= {key: scheduler_params.copy() for key in model}

    optimizer = build_optimizer({key: model[key].parameters() for key in model},
                                        scheduler_params_dict=scheduler_params_dict,
                                lr=float(config['optimizer_params'].get('lr', 1e-4)))
    
    for k, v in optimizer.optimizers.items():
        optimizer.optimizers[k] = accelerator.prepare(optimizer.optimizers[k])
        optimizer.schedulers[k] = accelerator.prepare(optimizer.schedulers[k])

    with accelerator.main_process_first():
        if config.get('pretrained_acoustic_model', '') != '':
            model, optimizer, start_epoch, iters = load_checkpoint(model,  optimizer, config['pretrained_acoustic_model'],
                                        load_only_params=config.get('load_only_params', True))
        else:
            start_epoch = 0
            iters = 0

    #sv_model = SVModel.from_pretrained("pyannote/wespeaker-voxceleb-resnet34-LM", 
    #                          use_auth_token="")
    #sv_model = sv_model.eval()

    gl = GeneratorLoss(model.md).to('cuda')
    dl = DiscriminatorLoss(model.md).to('cuda')
    wl = WavLMLoss(model.discriminator).to('cuda')
    sv = SVLoss().to('cuda')
    
    gl = accelerator.prepare(gl)
    dl = accelerator.prepare(dl)
    wl = accelerator.prepare(wl)
    sv = accelerator.prepare(sv)

    best_loss = float('inf')  # best test loss

    torch.cuda.empty_cache()

    stft_loss = MultiResolutionSTFTLoss().to('cuda')
    stft_loss = accelerator.prepare(stft_loss)
    
    loss_history = {
        'running_loss': deque(maxlen=log_interval),
        'loss_sty': deque(maxlen=log_interval),
        'd_loss': deque(maxlen=log_interval),
        'loss_algn': deque(maxlen=log_interval),
        'loss_s2s': deque(maxlen=log_interval),
        'loss_F0_rec': deque(maxlen=log_interval),
        'loss_gen_all': deque(maxlen=log_interval),
        'loss_lm': deque(maxlen=log_interval),
        'loss_gen_lm': deque(maxlen=log_interval),
        'fsim_loss': deque(maxlen=log_interval),
        'sim_loss': deque(maxlen=log_interval)
    }

    start_ds = True
    start_lm = True

    for epoch in range(start_epoch, epochs):
        running_loss = 0
        start_time = time.time()

        _ = [model[key].train() for key in model]
        
        for i, batch in enumerate(train_dataloader):
            waves = batch[0]
            batch = [b.to(device) if isinstance(b,list)==False else b for b in batch[1:]]
            texts, input_lengths, mels, mel_input_length, labels, ref_mels, ref_labels, adj_mels, adj_mels_lengths, ref_texts, ref_lengths = batch

            ### data preparation step

            with torch.no_grad():
                mask = length_to_mask(mel_input_length // (2 ** model.text_aligner.n_down)).to('cuda')
                text_mask = length_to_mask(input_lengths).to(texts.device)

            ppgs, s2s_pred, s2s_attn = model.text_aligner(mels, mask, texts)

            s2s_attn = s2s_attn.transpose(-1, -2)
            s2s_attn = s2s_attn[..., 1:]
            s2s_attn = s2s_attn.transpose(-1, -2)
            
            with torch.no_grad():
                attn_mask = (~mask).unsqueeze(-1).expand(mask.shape[0], mask.shape[1], text_mask.shape[-1]).float().transpose(-1, -2)
                attn_mask = attn_mask.float() * (~text_mask).unsqueeze(-1).expand(text_mask.shape[0], text_mask.shape[1], mask.shape[-1]).float()
                attn_mask = (attn_mask < 1)

            s2s_attn.masked_fill_(attn_mask, 0.0)
            
            with torch.no_grad():
                mask_ST = mask_from_lens(s2s_attn, input_lengths, mel_input_length // (2 ** model.text_aligner.n_down))
                s2s_attn_mono = maximum_path(s2s_attn, mask_ST)

            # encode
            t_en = model.text_encoder(texts, input_lengths, text_mask)
            
            mel_len_st = int(adj_mels_lengths.min().item() / 2 - 1)
            mel_len = min([int(mel_input_length.min().item() / 2 - 1)])

            st = []
            gt = []
            for bib in range(len(adj_mels_lengths)):
                mel_length = int(adj_mels_lengths[bib].item() / 2)
                random_start = np.random.randint(0, mel_length - mel_len_st)
                st.append(adj_mels[bib, :, (random_start * 2):((random_start+mel_len_st) * 2)])
                
                mel_length = int(mel_input_length[bib].item() / 2)
                random_start = np.random.randint(0, mel_length - mel_len)
                gt.append(mels[bib, :, (random_start * 2):((random_start+mel_len) * 2)])
                
            st = torch.stack(st).detach()
            gt = torch.stack(gt).detach()
            
            mmwd = False
            if random.random() < 1:
                st = gt
                mmwd = True
            
            s, t_en = model.style_encoder(st, t_en, input_lengths, t_en.size(-1))
            
            if random.random() < 0.2:
                with torch.no_grad():
                    s_null, t_en_null = model.style_encoder(torch.zeros_like(st).to('cuda'), t_en, input_lengths, t_en.size(-1))
                if bool(random.getrandbits(1)):
                    s = s_null
                else:
                    t_en = t_en_null

            if bool(random.getrandbits(1)):
                asr = (t_en @ s2s_attn)
            else:
                asr = (t_en @ s2s_attn_mono)

            # get clips
            mel_input_length_all = accelerator.gather(mel_input_length) # for balanced load
            mel_len = min([int(mel_input_length_all.min().item() / 2 - 1), 150])

            en = []
            gt = []
            wav = []
            
            rs = []
            
            for bib in range(len(mel_input_length)):
                mel_length = int(mel_input_length[bib].item() / 2)

                random_start = np.random.randint(0, mel_length - mel_len)
                rs.append(random_start)
                
                en.append(asr[bib, :, random_start:random_start+mel_len])
                gt.append(mels[bib, :, (random_start * 2):((random_start+mel_len) * 2)])
                
                y = waves[bib][(random_start * 2) * 300:((random_start+mel_len) * 2) * 300]
                wav.append(torch.from_numpy(y).to('cuda'))

            en = torch.stack(en)
            gt = torch.stack(gt).detach()
            wav = torch.stack(wav).float().detach()

            with torch.no_grad():
                # reconstruct
                real_norm = log_norm(gt.unsqueeze(1)).squeeze(1)

                F0_real = model.pitch_extractor(gt.unsqueeze(1))
        
            y_rec = model.decoder(en, F0_real, real_norm, s)

            loss_F0_rec = 0
            loss_sty = 0
        
            if start_ds:
                optimizer.zero_grad()
                d_loss = dl(wav.detach().unsqueeze(1).float(), y_rec.detach()).mean()
                if start_lm:
                    d_loss += wl(wav.detach().squeeze(), y_rec.detach().squeeze(), en.detach(), discriminator_turn=True).mean()
                accelerator.backward(d_loss)
                optimizer.step('md')
                optimizer.step('discriminator')

            else:
                d_loss = 0
            
            
            # generator loss
            optimizer.zero_grad()
            loss_mel = stft_loss(y_rec.squeeze(), wav.detach())

            loss_s2s = 0
            for _s2s_pred, _text_input, _text_length in zip(s2s_pred, texts, input_lengths):
                loss_s2s += F.cross_entropy(_s2s_pred[:_text_length], _text_input[:_text_length])
            loss_s2s /= texts.size(0)

            loss_algn = F.l1_loss(s2s_attn, s2s_attn_mono)
        
            
            if start_ds:
                loss_gen_all = gl(wav.detach().unsqueeze(1).float(), y_rec).mean()
            else:
                loss_gen_all = 0
                
            
            if start_ds:
                loss_gen_lm, loss_lm = wl(wav.detach().squeeze(), y_rec.squeeze(), en.detach(), generator_turn=True)
                loss_gen_lm, loss_lm = loss_gen_lm.mean(), loss_lm.mean()
            else:
                loss_lm, loss_gen_lm = 0, 0
            
            sim_loss, fsim_loss = sv(wav.unsqueeze(1).detach(), y_rec)
            
            sim_loss, fsim_loss = sim_loss.mean(), fsim_loss.mean()
            
            if start_ds:
                g_loss = loss_mel * 5 + loss_algn * 10 + loss_s2s + loss_gen_all + loss_lm + loss_gen_lm + sim_loss * 5 + fsim_loss * 5
            else:
                g_loss = loss_mel
                
            running_loss += accelerator.gather(loss_mel).mean().item()
            
            accelerator.backward(g_loss)

            if torch.isnan(g_loss):
                from IPython.core.debugger import set_trace
                set_trace()
            optimizer.step('text_encoder')
            optimizer.step('style_encoder')
            optimizer.step('decoder')
            if start_ds:
                optimizer.step('text_aligner')
                optimizer.step('pitch_extractor')

            # Add losses to history
            loss_history['running_loss'].append(loss_mel.item())
            loss_history['loss_sty'].append(loss_sty)
            loss_history['d_loss'].append(d_loss)
            loss_history['loss_algn'].append(loss_algn.item())
            loss_history['loss_s2s'].append(loss_s2s.item())
            loss_history['loss_F0_rec'].append(loss_F0_rec)
            loss_history['loss_gen_all'].append(loss_gen_all)
            loss_history['loss_lm'].append(loss_lm)
            loss_history['loss_gen_lm'].append(loss_gen_lm)
            loss_history['fsim_loss'].append(fsim_loss.item())
            loss_history['sim_loss'].append(sim_loss.item())
            
            iters = iters + 1

            if (i+1)%log_interval == 0 and accelerator.is_main_process:
                # Calculate average losses
                avg_losses = {k: sum(v)/len(v) if v else 0 for k, v in loss_history.items()}

                log_print ('Epoch [%d/%d], Step [%d/%d], Loss: %.5f, Sty Loss: %.5f, Disc Loss: %.5f, Algn Loss: %.5f, S2S Loss: %.5f, F0 Loss: %.5f, Gen Loss: %.5f, WavLM Loss: %.5f, GenLM Loss: %.5f, SIM Loss: %.5f, FSim Loss: %.5f'
                        %(epoch+1, epochs, i+1, len(train_dataloader), avg_losses["running_loss"], avg_losses["loss_sty"], avg_losses["d_loss"], avg_losses["loss_algn"], avg_losses["loss_s2s"], avg_losses["loss_F0_rec"], avg_losses["loss_gen_all"], avg_losses["loss_lm"], avg_losses["loss_gen_lm"], avg_losses["sim_loss"], avg_losses["fsim_loss"]), logger)
                
                writer.add_scalar('train/mel_loss', avg_losses["running_loss"], iters)
                writer.add_scalar('train/sty_loss', avg_losses["loss_sty"], iters)
                writer.add_scalar('train/d_loss', avg_losses["d_loss"], iters)
                writer.add_scalar('train/algn_loss', avg_losses["loss_algn"], iters)
                writer.add_scalar('train/s2s_loss', avg_losses["loss_s2s"], iters)
                writer.add_scalar('train/F0_rec_loss', avg_losses["loss_F0_rec"], iters)
                writer.add_scalar('train/gen_loss', avg_losses["loss_gen_all"], iters)
                writer.add_scalar('train/wavlm_loss', avg_losses["loss_lm"], iters)
                writer.add_scalar('train/genlm_loss', avg_losses["loss_gen_lm"], iters)
                writer.add_scalar('train/fsim_loss', avg_losses["fsim_loss"], iters)
                writer.add_scalar('train/sim_loss', avg_losses["sim_loss"], iters)
                
                running_loss = 0
                
                log_print('Time elapsed: %.2f seconds'%(time.time()-start_time),logger)
            
            if (i + 1) % 10000 == 0:
                loss_test = 0

                _ = [model[key].eval() for key in model]

                with torch.no_grad():
                    iters_test = 0
                    for batch_idx, batch in enumerate(val_dataloader):
                        optimizer.zero_grad()

                        waves = batch[0]
                        batch = [b.to(device) if isinstance(b,list)==False else b for b in batch[1:]]
                        texts, input_lengths, mels, mel_input_length, labels, ref_mels, ref_labels, adj_mels, adj_mels_lengths, ref_texts, ref_lengths = batch

                        with torch.no_grad():
                            mask = length_to_mask(mel_input_length // (2 ** model.text_aligner.n_down)).to('cuda')
                            ppgs, s2s_pred, s2s_attn = model.text_aligner(mels, mask, texts)

                            s2s_attn = s2s_attn.transpose(-1, -2)
                            s2s_attn = s2s_attn[..., 1:]
                            s2s_attn = s2s_attn.transpose(-1, -2)

                            text_mask = length_to_mask(input_lengths).to(texts.device)
                            attn_mask = (~mask).unsqueeze(-1).expand(mask.shape[0], mask.shape[1], text_mask.shape[-1]).float().transpose(-1, -2)
                            attn_mask = attn_mask.float() * (~text_mask).unsqueeze(-1).expand(text_mask.shape[0], text_mask.shape[1], mask.shape[-1]).float()
                            attn_mask = (attn_mask < 1)
                            s2s_attn.masked_fill_(attn_mask, 0.0)
                            mask_ST = mask_from_lens(s2s_attn, input_lengths, mel_input_length // (2 ** model.text_aligner.n_down))
                            s2s_attn_mono = maximum_path(s2s_attn, mask_ST)

                        # encode
                        t_en = model.text_encoder(texts, input_lengths, text_mask)

                        mel_len_st = int(mel_input_length.min().item() / 2 - 1)
                        st = []
                        for bib in range(len(mel_input_length)):
                            mel_length = int(mel_input_length[bib].item() / 2)
                            random_start = np.random.randint(0, mel_length - mel_len_st)
                            st.append(mels[bib, :, (random_start * 2):((random_start+mel_len_st) * 2)])
                        st = torch.stack(st).detach()

                        s, t_en = model.style_encoder(st, t_en, input_lengths, t_en.size(-1))

                        asr = (t_en @ s2s_attn)

                        # get clips
                        mel_input_length_all = accelerator.gather(mel_input_length) # for balanced load
                        mel_len = min([int(mel_input_length_all.min().item() / 2 - 1), 80])
                        en = []
                        gt = []
                        wav = []
                        for bib in range(len(mel_input_length)):
                            mel_length = int(mel_input_length[bib].item() / 2)

                            random_start = np.random.randint(0, mel_length - mel_len)
                            en.append(asr[bib, :, random_start:random_start+mel_len])
                            gt.append(mels[bib, :, (random_start * 2):((random_start+mel_len) * 2)])
                            y = waves[bib][(random_start * 2) * 300:((random_start+mel_len) * 2) * 300]
                            wav.append(torch.from_numpy(y).to('cuda'))

                        wav = torch.stack(wav).float().detach()

                        en = torch.stack(en)
                        gt = torch.stack(gt).detach()

                        F0_real = model.pitch_extractor(gt.unsqueeze(1))

                        real_norm = log_norm(gt.unsqueeze(1)).squeeze(1)
                        y_rec = model.decoder(en, F0_real, real_norm, s)

                        loss_mel = stft_loss(y_rec.squeeze(), wav.detach())

                        loss_test += loss_mel
                        iters_test += 1

                if accelerator.is_main_process:
                    print('Epochs:', epoch + 1)
                    print('Validation loss: %.3f' % (loss_test / iters_test), '\n\n\n')

                    if epoch % saving_epoch == 0:
                        if (loss_test / iters_test) < best_loss:
                            best_loss = loss_test / iters_test
                        print('Saving..')
                        state = {
                            'net':  {key: model[key].state_dict() for key in model},
                            'optimizer': optimizer.state_dict(),
                            'val_loss': loss_test / iters_test,
                            'epoch': epoch,
                            'iters': iters
                        }
                        if not os.path.isdir('acoustic_synth_checkpoint'):
                            os.mkdir('acoustic_synth_checkpoint')
                        torch.save(state, './acoustic_synth_checkpoint/val_loss_' + str((loss_test / iters_test)) + '.t7')

                _ = [model[key].train() for key in model]
                #asr_model = asr_model.eval()
                #F0_model = F0_model.eval()


if __name__=="__main__":
    main()