import numpy as np
import os
import pandas as pd
import time
import torch
from torch.nn.functional import log_softmax
from tqdm import tqdm

from lstm import GRUModel

LOG_2 = torch.log(torch.tensor(2.))


def sentence_predict(model, tokenizer, text, next_words=20):
    # note: model must not be on GPU
    x = torch.LongTensor(tokenizer.encode(text)).unsqueeze(0)
    model.eval()

    for _ in range(0, next_words):
        y_pred = model(x)

        last_word_logits = y_pred[0][-1]
        p = torch.nn.functional.softmax(last_word_logits, dim=0).detach().numpy()
        word_index = np.random.choice(len(last_word_logits), p=p)
        x = torch.cat((x,torch.LongTensor([[word_index]])), dim=1) # concat despite batch axis

    words = tokenizer.decode(x[0])
    return x, words


def test_predict_entropy(lm, dataloader, tokenizer, device, batch_predict_logits):
    tokens_logp = []
    sent_avg_logp = []
    iterator = tqdm(dataloader, desc='Iteration')
    for _, batch in enumerate(iterator):
        # call function
        batch_logp, batch_avg_logp = batch_predict_entropy(lm, batch, tokenizer, device, batch_predict_logits)
        tokens_logp.extend(batch_logp)
        sent_avg_logp.extend(batch_avg_logp)

    iterator.close()
    sent_length = [len(x) for x in tokens_logp]
    return sent_avg_logp, tokens_logp, sent_length

def batch_predict_logits_lm(lm, batch):
    with torch.no_grad():
        outputs = lm(**{'input_ids':batch['input_ids'], 'attention_mask':batch['attention_mask']})
    return outputs.logits

def batch_predict_logits_rnn(lm, batch):
    """Use with non HuggingFace model - otherwise other arguments in the dictionary will generate TypeError, 
    also return argument is a dict not an object
    """
    with torch.no_grad():
        outputs = lm(batch['input_ids'])
    return outputs



def batch_predict_entropy(lm, batch, tokenizer, device, batch_predict_logits): # might add logger here
    """ For one batch, get the entropy of expected words 

    Input:
        lm: model
        batch: dataloader batch with 'input_ids', 'attention_mask' keys, 'start_idx' (optional)
        batch_predict_logits: function, which predict function to use 
    """
    batch_length, max_sent_length = batch['input_ids'].shape
    for k in ['input_ids','attention_mask']:
        batch[k] = batch[k].to(device) # put data on gpu
    if 'start_idx' not in batch:
        batch['start_idx'] = [0 for _ in range(batch_length)]
    
    # returns
    batch_avg_logp = []
    batch_logp = []

    # get predictions
    lm.eval()
    outputs_logits = batch_predict_logits(lm, batch)
    logp_w = log_softmax(outputs_logits, dim=-1)
    logp_w /= LOG_2 # TODO: check

    # for every sentence
    for s_id in range(batch_length): 
        sentence = batch['input_ids'][s_id,:]
        sentence_logp = []
        # for every token
        for token_index in range(batch['start_idx'][s_id], max_sent_length): # -1 bc eos
            w_id = sentence[token_index]
            # skip special tokens (BOS, EOS, PAD) + speaker token # TODO: add speaker tokens
            if w_id in tokenizer.all_special_ids: # and w_id != unk_id:
                # print('w_id in tokenizer.all_special_ids')
                continue
            # increase non-normalised log probability of the sentence
            token_logp = logp_w[s_id, token_index, w_id].item()
            sentence_logp.append(token_logp)
        # append to batch
        batch_logp.append(sentence_logp)
        sentence_logp = np.array(sentence_logp)
        batch_avg_logp.append(- sentence_logp.sum()/sentence_logp.shape[0])
    
    return batch_logp, batch_avg_logp


def results_to_df(dataframe:pd.DataFrame, out_file_name:str,
                sent_avg_logp:list, tokens_logp:list, sent_length:list):
    # TODO: check all lengths
    dataframe['normalised_h'] = sent_avg_logp
    dataframe['length'] = sent_length
    dataframe['tokens_h'] = tokens_logp
    dataframe['sum_h'] = dataframe.normalised_h * dataframe.length
    # could add tokens to make sure which tokens

    h_bar = dataframe.groupby('length').agg({"normalised_h": "mean"}).to_dict()['normalised_h']
    dataframe['xu_h'] = dataframe.apply(lambda x: np.nan if x.length not in h_bar else x.normalised_h/h_bar[x.length], axis=1)

    dataframe.to_csv(f'{out_file_name}.csv',index=False)
    return dataframe