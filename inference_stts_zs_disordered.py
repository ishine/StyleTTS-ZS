idx = 4


import IPython.display as ipd

display(ipd.Audio(wav.detach().cpu()[idx].squeeze(), rate=24000))


display(ipd.Audio(y_rec.detach().cpu()[idx].squeeze(), rate=24000))


with torch.no_grad():
        iters_test = 0
        for batch_idx, batch in enumerate(val_dataloader):
                optimizer.zero_grad()

                waves = batch[0]
                batch = [b.to(device) for b in batch[1:]]
                texts, input_lengths, mels, mel_input_length, labels, ref_mels, ref_labels, adj_mels, adj_mels_lengths = batch
                iters_test += 1
                if iters_test > 3:
                    break


_ = [model[key].eval() for key in model]


idx = 8


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
    mask_ST = mask_from_lens(s2s_attn, input_lengths, mel_input_length // (2 ** asr_model.n_down))
    s2s_attn.masked_fill_(attn_mask, 0.0)
    s2s_attn_mono = maximum_path(s2s_attn, mask_ST)
    # encode
    t_en = model.text_encoder(texts, input_lengths, text_mask)
    asr = (t_en @ s2s_attn_mono)

    with torch.no_grad():
        F0_down = 7
        F0_real, _, F0 = F0_model(mels.unsqueeze(1))
        F0 = F0.reshape(F0.shape[0], F0.shape[1] * 2, F0.shape[2], 1).squeeze()
        F0_real = nn.functional.conv1d(F0_real.unsqueeze(1), torch.ones(1, 1, F0_down).to('cuda'), padding=F0_down//2).squeeze(1) / F0_down
        
    real_norm = log_norm(mels[idx, :, :mel_input_length[idx]].unsqueeze(0).unsqueeze(1)).squeeze(1)

    # reconstruct
    dix = idx
    
    with torch.no_grad():
        F0_ref, _, _ = F0_model(mels[(dix-1) % batch_size, :, :mel_input_length[(dix-1) % batch_size]].unsqueeze(0).unsqueeze(1))
        N_ref = log_norm(mels[(dix-1) % batch_size, :, :mel_input_length[(dix-1) % batch_size]].unsqueeze(0).unsqueeze(1)).squeeze(1)
        
        F0_ref_median = F0_ref.median()
        F0_trg = F0_real[idx, :mel_input_length[idx]].unsqueeze(0)
        F0_trg = F0_trg / F0_trg.median() * F0_ref_median
        
        N_ref_mean = N_ref.median()
        N_trg = real_norm / real_norm.median() * N_ref_mean
    
    
    s_trg, t_en_trg = model.style_encoder(mels[(dix-1) % batch_size, :, :mel_input_length[(dix-1) % batch_size] ].unsqueeze(0), 
                                t_en[idx, ...].unsqueeze(0), 
                               input_lengths[idx].unsqueeze(0),
                               t_en.size(-1))
    
    asr_trg = (t_en_trg @ s2s_attn_mono[idx, ...].unsqueeze(0))
    
    mel_fake = model.decoder(asr_trg[:, :, :mel_input_length[idx] // (2 ** asr_model.n_down)], 
                                F0_trg, N_trg,
                                s_trg)
    
    
    s, t_en_fake = model.style_encoder(adj_mels[idx, :, :adj_mels_lengths[idx] ].unsqueeze(0), 
                                t_en[idx, ...].unsqueeze(0), 
                               input_lengths[idx].unsqueeze(0),
                               t_en.size(-1))
    asr = (t_en_fake @ s2s_attn_mono[idx, ...].unsqueeze(0))

    y_rec_fake = model.decoder(asr[:, :, :mel_input_length[idx] // 2], 
                            F0_real[idx, :mel_input_length[idx]].unsqueeze(0), real_norm, 
                          s)
    
    s, t_en_asr = model.style_encoder(mels[idx, :, :mel_input_length[idx] ].unsqueeze(0), 
                                t_en[idx, ...].unsqueeze(0), 
                               input_lengths[idx].unsqueeze(0),
                               t_en.size(-1))
    
    asr = (t_en_asr @ s2s_attn_mono[idx, ...].unsqueeze(0))

    y_rec = model.decoder(asr[:, :, :mel_input_length[idx] // 2], 
                            F0_real[idx, :mel_input_length[idx]].unsqueeze(0), real_norm, 
                          s)


%matplotlib inline
import matplotlib.pyplot as plt
plt.imshow(s2s_attn[idx, :input_lengths[idx], :mel_input_length[idx] // 2].detach().cpu(), interpolation='none')
plt.ylabel('Phoneme index')
plt.xlabel('Mel-spectrogram frame index')
plt.title('Without TMA pre-training')


out = mel_fake

import IPython.display as ipd

display(ipd.Audio(mel_fake.cpu().squeeze()[..., 5000:-5000], rate=24000))

display(ipd.Audio(waves[dix-1].squeeze()[..., :-5000], rate=24000))


out = mel_fake

import IPython.display as ipd

display(ipd.Audio(mel_fake.cpu().squeeze()[..., 5000:-5000], rate=24000))

display(ipd.Audio(waves[dix-1].squeeze()[..., :-5000], rate=24000))


torch.allclose(adj_mels[idx, :, :adj_mels_lengths[idx] ], mels[idx, :, :mel_input_length[idx] ])


import IPython.display as ipd

display(ipd.Audio(y_rec.cpu().squeeze()[..., 5000:-5000], rate=24000))
display(ipd.Audio(y_rec_fake.cpu().squeeze()[..., 5000:-5000], rate=24000))


import IPython.display as ipd

display(ipd.Audio(y_rec.cpu().squeeze()[..., 5000:-5000], rate=24000))
display(ipd.Audio(y_rec_fake.cpu().squeeze()[..., 5000:-5000], rate=24000))


out = waves[idx]

import IPython.display as ipd
display(ipd.Audio(out, rate=24000))


with torch.no_grad():
    f0 = model.decoder.generator.f0_upsamp(F0_trg[:, None]).transpose(1, 2)
    har_source, _, uv = model.decoder.generator.m_source(f0)
    x_source = model.decoder.generator.noise_convs[-1](har_source.transpose(1, 2))


harm = (x_source.detach().cpu().squeeze(-1))[:, 11, :].squeeze()


outmap_min = harm.min()
outmap_max = harm.max()
harm = (harm - outmap_min) / (outmap_max - outmap_min) #


harm = harm - 0.5


display(ipd.Audio(harm.squeeze(), rate=24000))


plt.plot(harm[:, 2, :].squeeze())


with torch.no_grad():
    spec, phase = model.decoder.fw_phase(asr[idx, :, :mel_input_length[idx] // 2].unsqueeze(0), 
                        F0_real[idx, :mel_input_length[idx]].unsqueeze(0), real_norm, 
                      s)
    
    
plt.plot(spec.detach().cpu().squeeze()[0, 500:700])


plt.imshow(spec.detach().cpu().squeeze())


f, Pxx_den = signal.periodogram(waves[idx], 24000)