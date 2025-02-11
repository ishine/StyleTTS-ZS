import torch.nn as nn

class MultiTaskModel(nn.Module):
    def __init__(self, model, num_tokens=178, num_vocab=593, hidden_size=768):
        super().__init__()

        self.encoder = model
        self.mask_predictor = nn.Linear(hidden_size, num_tokens)
        self.word_predictor = nn.Linear(hidden_size, num_vocab)
    
    def forward(self, phonemes):
        output = self.encoder(phonemes)
        tokens_pred = self.mask_predictor(output.last_hidden_state)
        words_pred = self.word_predictor(output.last_hidden_state)
        
        return tokens_pred, words_pred