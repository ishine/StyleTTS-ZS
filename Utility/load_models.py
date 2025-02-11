import torch
from Modules.JDC.model import JDCNet
from Modules.ASR.models import ASRCNN
from Modules.BERT.model import MultiTaskModel
from Modules.rvq import ResidualVectorQuantize
from Modules.vocoder import Decoder
from Modules.encoders import TextEncoder, StyleEncoder, TVStyleEncoder
from Modules.discriminators import ProsodyDiscriminator, Discriminator
from Modules.predictors import DurationPredictor, ProsodyPredictor

from munch import Munch
import yaml
from transformers import AlbertConfig, AlbertModel

def load_F0_models(path):
    # load F0 model

    F0_model = JDCNet(num_class=1, seq_len=192)
    params = torch.load(path, weights_only=True)['net']
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


def load_bert(config_path):
    bert_config = yaml.safe_load(open(config_path))
    
    albert_base_configuration = AlbertConfig(**bert_config['model_params'])
    model = AlbertModel(albert_base_configuration)
    bert = MultiTaskModel(model).to('cuda')
    checkpoint = torch.load("Modules/BERT/step_1000000.t7", weights_only=True, map_location='cpu')
    state_dict = checkpoint['net']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    # load params
    bert.load_state_dict(new_state_dict, strict=False)
    print('Bert Checkpoint loaded.')
    return bert.encoder


def build_model(text_aligner, pitch_extractor, bert):
    decoder = Decoder(dim_in=512, F0_channel=512, style_dim=128, dim_out=80).to('cuda')
    text_encoder = TextEncoder(channels=512, kernel_size=5, depth=3, n_symbols=178).to('cuda')
    bert_encoder = torch.nn.Linear(768, 512).to('cuda')

    style_encoder = StyleEncoder(mel_dim=80, text_dim=512, style_dim=128, num_heads=8, num_layers=6).to('cuda')  
    #prosody_encoder = StyleEncoder(mel_dim=80, text_dim=512, style_dim=512, num_heads=8, num_layers=6).to('cuda')  

    dur_predictor = DurationPredictor(num_heads=8, head_features=64, d_hid=512, nlayers=6, max_dur=50).to('cuda')
    pro_predictor = ProsodyPredictor(num_heads=8, head_features=64, d_hid=512, nlayers=6).to('cuda')
    predictor_encoder = TVStyleEncoder(mel_dim=514, text_dim=512, 
                 num_heads=8, num_time=50, num_layers=6,
                 head_features=64).to('cuda')
    
    quantizer = ResidualVectorQuantize(
            input_dim=512, n_codebooks=9, codebook_dim=8
        ).to('cuda')
    
    text_embedding = torch.nn.Embedding(512, 512).to('cuda')
    
    pro_discriminator = ProsodyDiscriminator(mel_dim=514, num_heads=8, head_features=64, d_hid=512, nlayers=6).to('cuda')
    dur_discriminator = ProsodyDiscriminator(mel_dim=513, num_heads=8, head_features=64, d_hid=512, nlayers=6).to('cuda')
    md = Discriminator(sample_rate=24000).to('cuda')
    
    nets = Munch(bert=bert,
                 bert_encoder=bert_encoder,
                 text_embedding=text_embedding,
                 decoder=decoder,
                 pitch_extractor=pitch_extractor,
                 text_encoder=text_encoder,
                 style_encoder=style_encoder,
                 #prosody_encoder=prosody_encoder, # Not useful anymore, said the author
                 predictor_encoder=predictor_encoder,
                 dur_predictor=dur_predictor,
                 pro_predictor=pro_predictor,
                 quantizer=quantizer,
                 discriminator=pro_discriminator,
                 dur_discriminator=dur_discriminator,
                 text_aligner = text_aligner,
                 md = md
                 #mpd = MultiPeriodDiscriminator().to('cuda'),#
                 #msd = MultiResSpecDiscriminator().to('cuda')# Nunca se utilizan
                )

    return nets


def build_acoustic_model(text_aligner, pitch_extractor, training):
    decoder = Decoder(dim_in=512, F0_channel=512, style_dim=128, dim_out=80, training=training).to('cuda')
    text_encoder = TextEncoder(channels=512, kernel_size=5, depth=3, n_symbols=178).to('cuda')
    style_encoder = StyleEncoder(mel_dim=80, text_dim=512, style_dim=128, num_heads=8, num_layers=6).to('cuda')  
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
    state = torch.load(path, weights_only=True, map_location='cpu')
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