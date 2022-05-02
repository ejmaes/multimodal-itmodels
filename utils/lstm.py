import torch
import torch.nn as nn

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a tanh pre-decoder, a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers):
        self.predecoder_out = 400

        super(RNNModel, self).__init__()
        self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers)
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError("""An invalid option for `--model` was supplied,
                                      options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity)

        self.predecoder = nn.Linear(nhid, self.predecoder_out)
        self.tanh = nn.Tanh()
        self.decoder = nn.Linear(self.predecoder_out, ntoken)

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def forward(self, input, hidden):
        emb = self.encoder(input)
        output_rnn, hidden = self.rnn(emb, hidden)
        output_predecoder = self.tanh(self.predecoder(output_rnn))
        decoded = self.decoder(output_predecoder.view(output_predecoder.size(0)*output_predecoder.size(1), 
                                                output_predecoder.size(2)))
        return decoded.view(output_predecoder.size(0), output_predecoder.size(1), decoded.size(1)), hidden

class GRUModel(nn.Module):
    def __init__(self, tokenizer, embed_size=128, hidden_size=128, num_layers=1):
        super().__init__()
        self.embed = nn.Embedding(tokenizer.vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, hidden_size, num_layers=num_layers, bidirectional=False, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.decision = nn.Linear(hidden_size * 1 * 1, tokenizer.vocab_size)
        
    def forward(self, x, return_dict:bool=False):
        embed = self.embed(x)
        output, hidden = self.rnn(embed)
        drop = self.dropout(output)
        logits = self.decision(drop) # return batch, nb_words, tokenizer.vocab_size
        if not return_dict:
            return logits
        else:
            return {'logits': logits}