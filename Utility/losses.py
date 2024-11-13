import torch
import torchaudio
import torch.nn.functional as F
from pyannote.audio import Model as SVModel
from transformers import Wav2Vec2FeatureExtractor, WavLMModel, WavLMForXVector


sv_model = SVModel.from_pretrained("pyannote/wespeaker-voxceleb-resnet34-LM", 
                              use_auth_token="YOUR_AUTH_TOKEN")
sv_model = sv_model.eval()


wavlm = WavLMModel.from_pretrained('microsoft/wavlm-base-plus-sv').to('cuda')


def discriminator_TPRLS_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        tau = 0.04
        m_DG = torch.median((dr-dg))
        L_rel = torch.mean((((dr - dg) - m_DG)**2)[dr < dg + m_DG])
        loss += tau - F.relu(tau - L_rel)
    return loss

def generator_TPRLS_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    for dg, dr in zip(disc_real_outputs, disc_generated_outputs):
        tau = 0.04
        m_DG = torch.median((dr-dg))
        L_rel = torch.mean((((dr - dg) - m_DG)**2)[dr < dg + m_DG])
        loss += tau - F.relu(tau - L_rel)
    return loss


class GeneratorLoss(torch.nn.Module):

    def __init__(self, md):
        """Initilize spectral convergence loss module."""
        super(GeneratorLoss, self).__init__()
        self.md = md
        
    def forward(self, y, y_hat):
        d_fake = self.md(y_hat)
        d_real = self.md(y)
        
        loss_g = 0
        loss_rel = 0

        for x_fake, x_real in zip(d_fake, d_real):
            loss_g += torch.mean((1 - x_fake[-1]) ** 2)
#             loss_rel += generator_TPRLS_loss([x_real[-1]], [x_fake[-1]])

        loss_feature = 0
        for i in range(len(d_fake)):
            for j in range(len(d_fake[i]) - 1):
                loss_feature += F.l1_loss(d_fake[i][j], d_real[i][j].detach())

        
        loss_gen_all = loss_g + loss_feature + loss_rel
        
        return loss_gen_all.mean()
    
class DiscriminatorLoss(torch.nn.Module):

    def __init__(self, md):
        """Initilize spectral convergence loss module."""
        super(DiscriminatorLoss, self).__init__()
        self.md = md
        
    def forward(self, y, y_hat):
        d_fake = self.md(y_hat)
        d_real = self.md(y)
        loss_d = 0
        loss_rel = 0
        
        for x_fake, x_real in zip(d_fake, d_real):
            loss_d += torch.mean(x_fake[-1] ** 2)
            loss_d += torch.mean((1 - x_real[-1]) ** 2)
        
#             loss_rel += discriminator_TPRLS_loss([x_real[-1]], [x_fake[-1]])


        d_loss = loss_d + loss_rel
        
        return d_loss.mean()


class WavLMLoss(torch.nn.Module):

    def __init__(self, wavlm, mwd):
        """Initilize spectral convergence loss module."""
        super(WavLMLoss, self).__init__()
        self.wavlm = wavlm
        self.mwd = mwd
        self.resample = torchaudio.transforms.Resample(24000, 16000)
        
    def forward(self, wav, y_rec, text, generator_turn=False, discriminator_turn=False):
        assert generator_turn or discriminator_turn
        
        if generator_turn:
            return self.generator(wav, y_rec, text)
        if discriminator_turn:
            return self.discriminator(wav, y_rec, text)
        
    def generator(self, wav, y_rec, text, adv=True):
        with torch.no_grad():
            wav_16 = self.resample(wav)
            wav_embeddings = self.wavlm(input_values=wav_16, output_hidden_states=True).hidden_states
        y_rec_16 = self.resample(y_rec)
        y_rec_embeddings = self.wavlm(input_values=y_rec_16, output_hidden_states=True).hidden_states

        floss = 0
        for er, eg in zip(wav_embeddings, y_rec_embeddings):
            floss += torch.mean(torch.abs(er - eg))
    
        if adv:
            y_embeddings = torch.stack(wav_embeddings, dim=1).transpose(-1, -2).flatten(start_dim=1, end_dim=2)
            y_rec_embeddings = torch.stack(y_rec_embeddings, dim=1).transpose(-1, -2).flatten(start_dim=1, end_dim=2)

            with torch.no_grad():
                y_d_rs, r_hidden = self.mwd(y_embeddings, text, 
                         torch.ones(y_embeddings.size(0)).to(text.device).long() * y_embeddings.size(-1),
                        y_embeddings.size(-1))

            y_d_gs, f_hidden = self.mwd(y_rec_embeddings, text, 
                         torch.ones(y_rec_embeddings.size(0)).to(text.device).long() * y_rec_embeddings.size(-1),
                        y_rec_embeddings.size(-1))

            y_df_hat_r, y_df_hat_g = y_d_rs, y_d_gs

            loss_fm = 0
            for r, f in zip(r_hidden, f_hidden):
                loss_fm = F.l1_loss(r, f)

            loss_gen_f = torch.mean((1-y_df_hat_g)**2)
    #         loss_rel = generator_TPRLS_loss([y_df_hat_r], [y_df_hat_g])
            loss_rel = 0

            loss_gen_all = loss_gen_f + loss_rel + loss_fm
        else:
            loss_gen_all = torch.zeros(1).to(wav.device)
            
        return loss_gen_all, floss.mean()
    
    def discriminator(self, wav, y_rec, text):
        with torch.no_grad():
            wav_16 = self.resample(wav)
            wav_embeddings = self.wavlm(input_values=wav_16, output_hidden_states=True).hidden_states
            y_rec_16 = self.resample(y_rec)
            y_rec_embeddings = self.wavlm(input_values=y_rec_16, output_hidden_states=True).hidden_states

            y_embeddings = torch.stack(wav_embeddings, dim=1).transpose(-1, -2).flatten(start_dim=1, end_dim=2)
            y_rec_embeddings = torch.stack(y_rec_embeddings, dim=1).transpose(-1, -2).flatten(start_dim=1, end_dim=2)

        y_d_rs, _ = self.mwd(y_embeddings, text, 
                     torch.ones(y_embeddings.size(0)).to(text.device).long() * y_embeddings.size(-1),
                    y_embeddings.size(-1))
        
        y_d_gs, _ = self.mwd(y_rec_embeddings, text, 
                     torch.ones(y_rec_embeddings.size(0)).to(text.device).long() * y_rec_embeddings.size(-1),
                    y_rec_embeddings.size(-1))
        
        y_df_hat_r, y_df_hat_g = y_d_rs, y_d_gs
        
        r_loss = torch.mean((1-y_df_hat_r)**2)
        g_loss = torch.mean((y_df_hat_g)**2)
        
        loss_disc_f = r_loss + g_loss
        
#         loss_rel = discriminator_TPRLS_loss([y_df_hat_r], [y_df_hat_g])
        loss_rel = 0
    
        d_loss = loss_disc_f + loss_rel
        
        return d_loss.mean()


class SVLoss(torch.nn.Module):

    def __init__(self, sv_model):
        """Initilize spectral convergence loss module."""
        super(SVLoss, self).__init__()
        self.resample = torchaudio.transforms.Resample(24000, 16000)
        self.sv_model = sv_model
        
    def forward(self, y, y_hat):
        y_hat = self.resample(y_hat)
        h_fake, emb_fake = self.sv_model(y_hat, return_features=True)
        
        with torch.no_grad():
            y = self.resample(y)
            h_real, emb_real = self.sv_model(y, return_features=True)
        
        loss_feat = 0
        for x_fake, x_real in zip(h_fake, h_real):
            loss_feat += F.l1_loss(x_fake, x_real)
        
        loss_sim = 1 - F.cosine_similarity(emb_fake, emb_real, dim=-1).mean()
        
        return loss_feat, loss_sim


class SpectralConvergengeLoss(torch.nn.Module):
    """Spectral convergence loss module."""

    def __init__(self):
        """Initilize spectral convergence loss module."""
        super(SpectralConvergengeLoss, self).__init__()

    def forward(self, x_mag, y_mag):
        """Calculate forward propagation.
        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
        Returns:
            Tensor: Spectral convergence loss value.
        """
        return torch.norm(y_mag - x_mag, p=1) / torch.norm(y_mag, p=1)

class STFTLoss(torch.nn.Module):
    """STFT loss module."""

    def __init__(self, fft_size=1024, shift_size=120, win_length=600, window=torch.hann_window):
        """Initialize STFT loss module."""
        super(STFTLoss, self).__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.to_mel = torchaudio.transforms.MelSpectrogram(sample_rate=24000, n_fft=fft_size, win_length=win_length, hop_length=shift_size, window_fn=window)

        self.spectral_convergenge_loss = SpectralConvergengeLoss()

    def forward(self, x, y):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
        Returns:
            Tensor: Spectral convergence loss value.
            Tensor: Log STFT magnitude loss value.
        """
        x_mag = self.to_mel(x)
        mean, std = -4, 4
        x_mag = (torch.log(1e-5 + x_mag) - mean) / std
        
        y_mag = self.to_mel(y)
        mean, std = -4, 4
        y_mag = (torch.log(1e-5 + y_mag) - mean) / std
        
        sc_loss = self.spectral_convergenge_loss(x_mag, y_mag)

        return sc_loss


class MultiResolutionSTFTLoss(torch.nn.Module):
    """Multi resolution STFT loss module."""

    def __init__(self,
                 fft_sizes=[1024, 2048, 512],
                 hop_sizes=[120, 240, 50],
                 win_lengths=[600, 1200, 240],
                 window=torch.hann_window):
        """Initialize Multi resolution STFT loss module.
        Args:
            fft_sizes (list): List of FFT sizes.
            hop_sizes (list): List of hop sizes.
            win_lengths (list): List of window lengths.
            window (str): Window function type.
        """
        super(MultiResolutionSTFTLoss, self).__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.stft_losses = torch.nn.ModuleList()
        for fs, ss, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.stft_losses += [STFTLoss(fs, ss, wl, window)]

    def forward(self, x, y):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
        Returns:
            Tensor: Multi resolution spectral convergence loss value.
            Tensor: Multi resolution log STFT magnitude loss value.
        """
        sc_loss = 0.0
        for f in self.stft_losses:
            sc_l = f(x, y)
            sc_loss += sc_l
        sc_loss /= len(self.stft_losses)

        return sc_loss
