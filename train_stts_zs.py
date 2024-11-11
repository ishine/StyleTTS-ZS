import torch
import time
import random
import torch.nn as nn
import os
from os import listdir
from os.path import isfile, join
import os.path as osp
from Dataset.meldataset import build_dataloader
from Utility.optimizers import build_optimizer
from Utility.utils import *
from Utility.load_models import *
from Utility.losses import *
import click
import yaml
import shutil

from accelerate import Accelerator
from accelerate.utils import LoggerType
from accelerate import DistributedDataParallelKwargs

from torch.utils.tensorboard import SummaryWriter

# simple fix for dataparallel that allows access to class attributes
class MyDataParallel(torch.nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

import logging
from accelerate.logging import get_logger
logger = get_logger(__name__, log_level="DEBUG")

@click.command()
@click.option('-p', '--config_path', default='Configs/config.yml', type=str)
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
    file_handler = logging.FileHandler(osp.join(log_dir, 'train.log'))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(levelname)s:%(asctime)s: %(message)s'))
    logger.logger.addHandler(file_handler)

    batch_size = config.get('batch_size', 16)
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
                                        num_workers=8,
                                        dataset_config={},
                                        device=device)

    val_dataloader = build_dataloader(val_list,
                                    batch_size=batch_size,
                                    validation=True,
                                    num_workers=2,
                                    device=device,
                                    dataset_config={})
    
    with accelerator.main_process_first():
         # load pretrained ASR model
        ASR_config = config.get('ASR_config', False)
        ASR_path = config.get('ASR_path', False)
        asr_model = load_ASR_models(ASR_path, ASR_config)

        # load pretrained F0 model
        F0_path = config.get('F0_path', False)
        F0_model = load_F0_models(F0_path)

    # i, (waves, texts, input_lengths, mels, mel_input_length, labels, ref_mels, ref_labels, adj_mels, adj_mels_lengths) = next(enumerate(train_dataloader))

    model = build_model(asr_model, F0_model)    

    with accelerator.main_process_first():
        if config.get('pretrained_model', '') != '':
            model, optimizer, start_epoch, iters = load_checkpoint(model,  optimizer, config['pretrained_model'],
                                        load_only_params=config.get('load_only_params', True))
        else:
            start_epoch = 0
            iters = 0

    for key in model:
        if key != "md" and key != "discriminator":
            model[key] = MyDataParallel(model[key])

    gl = GeneratorLoss(model.md).to('cuda')
    dl = DiscriminatorLoss(model.md).to('cuda')
    wl = WavLMLoss(wavlm, model.discriminator).to('cuda')
    sv = SVLoss(sv_model).to('cuda')

    gl = MyDataParallel(gl)
    dl = MyDataParallel(dl)
    wl = MyDataParallel(wl)
    sv = MyDataParallel(sv)
    
    
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

    best_loss = float('inf')  # best test loss
    loss_train_record = list([])
    loss_test_record = list([])

    criterion = nn.L1Loss() # F0 loss (regression)
    torch.cuda.empty_cache()

    stft_loss = MultiResolutionSTFTLoss().to('cuda')

    mypath  ="/local/data_cache/"

    start_ds = True
    start_lm = True

    coll = 0

    for epoch in range(start_epoch, epochs):
        running_loss = 0
        start_time = time.time()

        _ = [model[key].train() for key in model]
        
        for i, batch in enumerate(train_dataloader):
            onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
            cache_file = mypath + random.choice(onlyfiles)
            try:
                batch = torch.load(cache_file)
            except:
                os.remove(cache_file)
                continue
                
            libri = False
                
            if len(onlyfiles) > 100 and (i + 1) % 50 != 0:
                try:
                    os.remove(cache_file)
                except:
                    pass
                waves = batch[0]
                batch = batch[1:]

                batch = [b.to(device) for b in batch]
                texts, input_lengths, mels, mel_input_length, adj_texts, adj_input_lengths, adj_mels, adj_mels_lengths = batch
                libri = True

            else:            
                _, batch = next(enumerate(train_dataloader))
                waves = batch[0]
                batch = [b.to(device) for b in batch[1:]]
                texts, input_lengths, mels, mel_input_length, labels, ref_mels, ref_labels, adj_mels, adj_mels_lengths = batch
                libri = False

            
            ### data preparation step
            
            with torch.no_grad():
                mask = length_to_mask(mel_input_length // (2 ** asr_model.n_down)).to('cuda')
                text_mask = length_to_mask(input_lengths).to(texts.device)

            ppgs, s2s_pred, s2s_attn = asr_model(mels, mask, texts)

            s2s_attn = s2s_attn.transpose(-1, -2)
            s2s_attn = s2s_attn[..., 1:]
            s2s_attn = s2s_attn.transpose(-1, -2)
            
            with torch.no_grad():
                attn_mask = (~mask).unsqueeze(-1).expand(mask.shape[0], mask.shape[1], text_mask.shape[-1]).float().transpose(-1, -2)
                attn_mask = attn_mask.float() * (~text_mask).unsqueeze(-1).expand(text_mask.shape[0], text_mask.shape[1], mask.shape[-1]).float()
                attn_mask = (attn_mask < 1)

            s2s_attn.masked_fill_(attn_mask, 0.0)
            
            with torch.no_grad():
                mask_ST = mask_from_lens(s2s_attn, input_lengths, mel_input_length // (2 ** asr_model.n_down))
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
                
                if not libri:
                    y = waves[bib][(random_start * 2) * 300:((random_start+mel_len) * 2) * 300]
                    wav.append(torch.from_numpy(y).to('cuda'))
                else:
                    y = waves.squeeze()[bib][(random_start * 2) * 300:((random_start+mel_len) * 2) * 300]
                    wav.append((y).to('cuda'))


            en = torch.stack(en)
            gt = torch.stack(gt).detach()
            wav = torch.stack(wav).float().detach()

            with torch.no_grad():
                _, F0_gt, _ = F0_model(gt.unsqueeze(1))
                
                # reconstruct
                real_norm = log_norm(gt.unsqueeze(1)).squeeze(1)

                F0_real, _, _ = F0_model(gt.unsqueeze(1))
        
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
                g_loss = loss_mel * 5 + loss_sty + loss_algn * 10 + loss_s2s + loss_F0_rec + loss_gen_all + loss_lm + loss_gen_lm + sim_loss * 5 + fsim_loss * 5
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

            iters = iters + 1

            if (i+1)%log_interval == 0 and accelerator.is_main_process:
                log_print ('Epoch [%d/%d], Step [%d/%d], Loss: %.5f, Sty Loss: %.5f, Disc Loss: %.5f, Algn Loss: %.5f, S2S Loss: %.5f, F0 Loss: %.5f, Gen Loss: %.5f, WavLM Loss: %.5f, GenLM Loss: %.5f, SIM Loss: %.5f, FSim Loss: %.5f'
                        %(epoch+1, epochs, i+1, len(train_dataloader)//batch_size, running_loss / log_interval, loss_sty, d_loss, loss_algn, loss_s2s, loss_F0_rec, loss_gen_all, loss_lm, loss_gen_lm, fsim_loss, sim_loss), logger)
                
                writer.add_scalar('train/mel_loss', running_loss / log_interval, iters)
                writer.add_scalar('train/sty_loss', loss_sty, iters)
                writer.add_scalar('train/d_loss', d_loss, iters)
                writer.add_scalar('train/algn_loss', loss_algn, iters)
                writer.add_scalar('train/s2s_loss', loss_s2s, iters)
                writer.add_scalar('train/F0_rec_loss', loss_F0_rec, iters)
                writer.add_scalar('train/gen_loss', loss_gen_all, iters)
                writer.add_scalar('train/lm_loss', loss_lm, iters)
                writer.add_scalar('train/gen_lm_loss', loss_gen_lm, iters)
                writer.add_scalar('train/fsim_loss', fsim_loss, iters)
                writer.add_scalar('train/sim_loss', sim_loss, iters)
                
                running_loss = 0
                
                print('Time elasped:', time.time()-start_time)
            
            if (i + 1) % 10000 == 0:
                loss_test = 0

                _ = [model[key].eval() for key in model]

                with torch.no_grad():
                    iters_test = 0
                    for batch_idx, batch in enumerate(val_dataloader):
                        optimizer.zero_grad()

                        waves = batch[0]
                        batch = [b.to(device) for b in batch[1:]]
                        texts, input_lengths, mels, mel_input_length, labels, ref_mels, ref_labels, adj_mels, adj_mels_lengths = batch

                        with torch.no_grad():
                            mask = length_to_mask(mel_input_length // (2 ** asr_model.n_down)).to('cuda')
                            ppgs, s2s_pred, s2s_attn = asr_model(mels, mask, texts)

                            s2s_attn = s2s_attn.transpose(-1, -2)
                            s2s_attn = s2s_attn[..., 1:]
                            s2s_attn = s2s_attn.transpose(-1, -2)

                            text_mask = length_to_mask(input_lengths).to(texts.device)
                            attn_mask = (~mask).unsqueeze(-1).expand(mask.shape[0], mask.shape[1], text_mask.shape[-1]).float().transpose(-1, -2)
                            attn_mask = attn_mask.float() * (~text_mask).unsqueeze(-1).expand(text_mask.shape[0], text_mask.shape[1], mask.shape[-1]).float()
                            attn_mask = (attn_mask < 1)
                            s2s_attn.masked_fill_(attn_mask, 0.0)
                            mask_ST = mask_from_lens(s2s_attn, input_lengths, mel_input_length // (2 ** asr_model.n_down))
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

                        F0_real, _, F0 = F0_model(gt.unsqueeze(1))
                        F0 = F0.reshape(F0.shape[0], F0.shape[1] * 2, F0.shape[2], 1).squeeze()

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
                        }
                        if not os.path.isdir('checkpoint_PH_E2E_real_fixed_exactF0_gtstyle_snake_R_newsine_conf_dropout_F0_librilight_gt'):
                            os.mkdir('checkpoint_PH_E2E_real_fixed_exactF0_gtstyle_snake_R_newsine_conf_dropout_F0_librilight_gt')
                        torch.save(state, './checkpoint_PH_E2E_real_fixed_exactF0_gtstyle_snake_R_newsine_conf_dropout_F0_librilight_gt/val_loss_' + str((loss_test / iters_test)) + '.t7')

                _ = [model[key].train() for key in model]
                asr_model = asr_model.eval()
                F0_model = F0_model.eval()

    # loss_test, iters_test = 0, 1

if __name__=="__main__":
    main()