import torch
from Modules.JDC.model import JDCNet
from Modules.ASR.models import ASRCNN

from Modules.vocoder import Decoder
from Modules.encoders import TextEncoder, TVStyleEncoder
from Modules.discriminators import ProsodyDiscriminator, Discriminator

from munch import Munch
import yaml


def load_F0_models(path):
    # load F0 model

    F0_model = JDCNet(num_class=1, seq_len=192)
    params = torch.load(path)['net']
    F0_model.load_state_dict(params)
    _ = F0_model.train()
    F0_model = F0_model.to('cuda')

    return F0_model


def load_ASR_models(ASR_MODEL_PATH, ASR_MODEL_CONFIG):
    # load ASR model

    def _load_config(path):
        with open(path) as f:
            config = yaml.safe_load(f)
        model_config = config['model_params']
        return model_config

    def _load_model(model_config, model_path):
        model = ASRCNN(**model_config)
        params = torch.load(model_path, map_location='cpu')['model']
        model.load_state_dict(params)
        return model

    asr_model_config = _load_config(ASR_MODEL_CONFIG)
    asr_model = _load_model(asr_model_config, ASR_MODEL_PATH)
    _ = asr_model.train()
    asr_model = asr_model.to('cuda')

    return asr_model


def build_model(text_aligner, pitch_extractor):
    decoder = Decoder(dim_in=512, F0_channel=512, style_dim=128, dim_out=80).to('cuda')
    text_encoder = TextEncoder(channels=512, kernel_size=5, depth=3, n_symbols=178).to('cuda')
    style_encoder = TVStyleEncoder(mel_dim=80, text_dim=512, style_dim=128, num_heads=8, num_layers=6).to('cuda')  
    discriminator = ProsodyDiscriminator(mel_dim=768 * 13, num_heads=8, head_features=64, d_hid=512, nlayers=6).to('cuda')
    md = Discriminator(sample_rate=24000).to('cuda')  
    
    nets = Munch(decoder=decoder,
                 pitch_extractor=pitch_extractor,
                 text_encoder=text_encoder,
                 style_encoder=style_encoder,
                 text_aligner = text_aligner,
                 discriminator=discriminator,
                 md=md,
                )

    return nets


def load_checkpoint(model, optimizer, path, load_only_params=True, ignore_modules=[]):
    state = torch.load(path, map_location='cpu')
    params = state['net']
    for key in model:
        if key in params and key not in ignore_modules:
            print('%s loaded' % key)
            try:
                model[key].load_state_dict(params[key])
            except:
                from collections import OrderedDict
                state_dict = params[key]
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:] # remove `module.`
                    new_state_dict[name] = v
                # load params
                model[key].load_state_dict(new_state_dict)
    _ = [model[key].eval() for key in model]
    
    if not load_only_params:
        epoch = state["epoch"]
        iters = state["iters"]
        optimizer.load_state_dict(state["optimizer"])
    else:
        epoch = 0
        iters = 0
        
    return model, optimizer, epoch, iters