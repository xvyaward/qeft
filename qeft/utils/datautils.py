import numpy as np
import random
import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, LlamaTokenizer
from typing import Union

import datasets
import pandas as pd
import re

def get_wikitext2(nsamples, seed, seqlen, tokenizer, train):
    if train:
        traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
        trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
        
        random.seed(seed)
        trainloader = []
        for _ in range(nsamples):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
            
        return trainloader
            
    else:
        testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
        testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')
        
        return testenc

def get_ptb(nsamples, seed, seqlen, tokenizer, train):
    if train:
        traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
        trainenc = tokenizer(" ".join(traindata['sentence']), return_tensors='pt')

        random.seed(seed)
        trainloader = []
        for _ in range(nsamples):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
            
        return trainloader
    else:
        testdata = load_dataset("ptb_text_only", "penn_treebank", split="test")
        testenc = tokenizer(" ".join(testdata["sentence"]), return_tensors="pt")
        
        return testenc

def get_c4(nsamples, seed, seqlen, tokenizer, train):
    if train:
        traindata = load_dataset(
            'allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
        )
        random.seed(seed)
        trainloader = []
        for _ in range(nsamples):
            while True:
                i = random.randint(0, len(traindata) - 1)
                trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
                if trainenc.input_ids.shape[1] > seqlen:
                    break
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
        
        return trainloader
    else:
        valdata = load_dataset(
            'allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation'
        )
        valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
        valenc = valenc.input_ids[:, :(256 * seqlen)]

        class TokenizerWrapper:
            def __init__(self, input_ids):
                self.input_ids = input_ids
        valenc = TokenizerWrapper(valenc)

        return valenc

def get_platypus(nsamples, seed, seqlen, tokenizer, train):
    if train:
        traindata = load_dataset(
            "garage-bAInd/Open-Platypus", split='train', 
        )
        #import code; code.interact("platypus check", local=locals())
        
        concatenated_data = [a + ' ' + b for a, b in zip(traindata['instruction'], traindata['output'])]
        trainenc = tokenizer(" ".join(concatenated_data), return_tensors='pt')

        random.seed(seed)
        trainloader = []
        for _ in range(nsamples):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
        
        del traindata
        del concatenated_data
        del trainenc

        #import code; code.interact("check platypus trainloader", local=locals())
        
        return trainloader


def preprocess(text):
    text = text.strip()
    # NOTE: Brackets are artifacts of the WikiHow dataset portion of HellaSwag.
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text

def _process_doc(doc):
    ctx = doc["ctx_a"] + " " + doc["ctx_b"].capitalize()
    out_doc = {
        "ctx": preprocess(ctx),
        "query": preprocess(doc["activity_label"] + ": " + ctx),
        "choices": [preprocess(ending) for ending in doc["endings"]],
        "gold": int(doc["label"]),
    }
    return out_doc

def get_hellaswag(nsamples, seed, seqlen, tokenizer, train):
    #import code; code.interact("hellaswag", local=locals())
    if train:
        traindata = load_dataset("hellaswag", split='train')
        _training_docs = list(map(_process_doc, traindata))
        # len(_training_docs) = 39905
        post_process_data = datasets.Dataset.from_pandas(pd.DataFrame(data=_training_docs))
        trainenc = tokenizer(" ".join(post_process_data['ctx']), return_tensors='pt')

        #import code; code.interact("hellaswag check", local=locals())

        random.seed(seed)
        trainloader = []
        for _ in range(nsamples):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
        
        del post_process_data
        del trainenc
        return trainloader

    else:
        raise NotImplementedError

def get_loaders(
    name, nsamples=128, seed=0, seqlen=2048, model='', train=True
):
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, tokenizer, train)
    elif 'ptb' in name:
        return get_ptb(nsamples, seed, seqlen, tokenizer, train)
    elif 'c4' in name:
        return get_c4(nsamples, seed, seqlen, tokenizer, train)
    ## added
    elif 'platypus' in name:
        tokenizer = LlamaTokenizer.from_pretrained(model)
        tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
        tokenizer.padding_side = "right"
        return get_platypus(nsamples, seed, seqlen, tokenizer, train)
    elif 'hellaswag' in name:
        return get_hellaswag(nsamples, seed, seqlen, tokenizer, train)
    else: # custom dataset
        print(f"Custom dataset load from {name}")
        datas = torch.load(name)
        ids_shuffle = list(range(len(datas)))
        random.shuffle(ids_shuffle)
        return [tuple(datas[idx].unsqueeze(0)) for idx in ids_shuffle[:nsamples]]