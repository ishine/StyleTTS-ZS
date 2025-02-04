import torch
import torchaudio
from monotonic_align import mask_from_lens
from Utility.utils import *
import soundfile as sf
import librosa
import yaml
from Utility.load_models import *
from phonemizer import phonemize


_pad = "$"
_punctuation = ';:,.!?¡¿—…"«»“” '
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
_letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"

# Export all symbols:
symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)

dicts = {}
for i in range(len((symbols))):
    dicts[symbols[i]] = i

class TextCleaner:
    def __init__(self, dummy=None):
        self.word_index_dictionary = dicts
        #print(len(dicts))
    def __call__(self, text):
        indexes = []
        for char in text:
            try:
                indexes.append(self.word_index_dictionary[char])
            except KeyError:
                pass
                #print(text)
        return indexes

to_mel = torchaudio.transforms.MelSpectrogram(
    n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
mean, std = -4, 4

def preprocess(wave):
    wave_tensor = torch.from_numpy(wave).float()
    mel_tensor = to_mel(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
    return mel_tensor

def inference(model, mels, mel_input_length, texts, input_lengths):
    _ = [model[key].eval() for key in model]
    
    mels = mels.to('cuda')
    mel_input_length = mel_input_length.to('cuda')
    texts = texts.to('cuda')
    input_lengths = input_lengths.to('cuda')
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
        #mask_ST = mask_from_lens(s2s_attn, input_lengths, mel_input_length // (2 ** model.text_aligner.n_down))
        #s2s_attn_mono = maximum_path(s2s_attn, mask_ST)

        # encode
        t_en = model.text_encoder(texts, input_lengths, text_mask)

        #mel_len_st = int(mel_input_length.min().item() / 2 - 1)
        st = []
        for bib in range(len(mel_input_length)):
            mel_length = int(mel_input_length[bib].item())
            #random_start = np.random.randint(0, mel_length - mel_len_st)
            st = mels[bib, :, :mel_length].unsqueeze(0)
        #st = torch.stack(st).detach()

        s, t_en = model.style_encoder(st, t_en, input_lengths, t_en.size(-1))

        asr = (t_en @ s2s_attn)

        # get clips
        #mel_len = min([int(mel_input_length.min().item() / 2 - 1), 80])
        #en = []
        #gt = []
        for bib in range(len(mel_input_length)):
            mel_length = int(mel_input_length[bib].item())

            #random_start = np.random.randint(0, mel_length - mel_len)
            en = asr[bib, :, :mel_length // 2].unsqueeze(0)
            gt = mels[bib, :, :mel_length].unsqueeze(0)

        #en = torch.stack(en)
        #gt = torch.stack(gt).detach()

        F0_real = model.pitch_extractor(gt.unsqueeze(1))
        
        real_norm = log_norm(gt.unsqueeze(1)).squeeze(1)
        y_rec = model.decoder(en, F0_real, real_norm, s)
        
        sf.write("output.wav", y_rec.cpu().squeeze()[..., 5000:-5000], 24000)

if __name__ == "__main__":
    wave, sr = sf.read("multiplepause.wav")
    if wave.shape[-1] == 2:
        wave = wave[:, 0].squeeze()
    if sr != 24000:
        wave = librosa.resample(wave, orig_sr=sr, target_sr=24000)
        #print(wave_path, sr)
    wave = np.concatenate([np.zeros([5000]), wave, np.zeros([5000])], axis=0)
    
    mel_tensor = preprocess(wave).squeeze()
    
    mel_tensor = mel_tensor.squeeze()
    length_feature = mel_tensor.size(1)
    mel_tensor = mel_tensor[:, :(length_feature - length_feature % 2)]
    
    nmels = mel_tensor.size(0)
    max_mel_length = max([mel_tensor.shape[1]])
    mels = torch.zeros((1, nmels, max_mel_length)).float()
    mel_size = mel_tensor.size(1)
    mels[0, :, :mel_size] = mel_tensor    
    
    mel_input_length = torch.zeros(1).long()
    mel_input_length[0] = mel_size
    
    text = " But I don't know how to...Uhmm...Maybe you're right...Ahhh...We could do it that way...Yeah! Why not?"
    text = phonemize(
        text,
        language="en-us",
        backend="espeak",
        preserve_punctuation=True,
        with_stress=True,
        language_switch='remove-flags'
    )
    transphorm = {'œ̃': 'ʘ', 'ɑ̃': 'ɺ', 'ɔ̃': 'ɻ', 'ɛ̃': 'ʀ', 'a-': 'ʧ', 'ə-': 'ʉ', 'y-': 'ʋ', 'e-': 'ⱱ'}
    for phoneme, replacement in transphorm.items():
        text = text.replace(phoneme, replacement)
    text_cleaner = TextCleaner()
    text = text_cleaner(text)
        
    text.insert(0, 0)
    text.append(0)
    
    text = torch.LongTensor(text)
    max_text_length = max([text.shape[0]])
    texts = torch.zeros((1, max_text_length)).long()
    
    text_size = text.size(0)
    texts[0, :text_size] = text
    
    input_lengths = torch.zeros(1).long()
    input_lengths[0] = text_size
    
    config = yaml.safe_load(open('Configs/config_acoustic.yml'))
    ASR_config = config.get('ASR_config', False)
    ASR_path = config.get('ASR_path', False)
    asr_model = load_ASR_models(ASR_path, ASR_config)

    # load pretrained F0 model
    F0_path = config.get('F0_path', False)
    F0_model = load_F0_models(F0_path)
    
    model = build_acoustic_model(asr_model, F0_model, training=False)
    model, _, _, _ = load_checkpoint(model,  None, config['pretrained_acoustic_model'],
                                     load_only_params=config.get('load_only_params', True))
    
    inference(model, mels, mel_input_length, texts, input_lengths)