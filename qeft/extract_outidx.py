import time

import torch
import torch.nn as nn

import argparse
import numpy as np
from tqdm import tqdm
import os

from qeft.recon import GPTQ_OWQ
from qeft.quant import *
from qeft.utils.misc import *
from qeft.utils.datautils import *
from qeft.utils.modelutils import *
from qeft.reorder import *

@torch.no_grad()
def extract_outlieridx(model, dataloader, dev, args):
    if args.perhead is not None:
        args.target_rank = args.perhead * model.config.num_attention_heads
        
    dirname = os.path.join(args.output_dir, f"w{args.wbits}_r{args.target_rank}")
    if not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)
        
    d = model.get_input_embeddings().weight.shape[1]
    sensitivity_sum = torch.zeros([d], dtype=torch.float, device=dev)
    
    meta = args.meta
    print('Starting ...')

    use_cache = model.config.use_cache
    layers, pre_layers, _ = parsing_layers(model, meta)
    model.config.use_cache = False
    
    for pre_layer in pre_layers:
        pre_layer = pre_layer.to(dev)
    
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, args.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )

    cache = {kw:None for kw in meta['inp_kwargs']}
    cache['i'] = 0

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            for key in cache:
                if key == 'i':
                    cache['i'] += 1
                else:
                    cache[key] = kwargs[key]
            raise ValueError
    
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    
    layers[0] = layers[0].module.cpu()
    
    for pre_layer in pre_layers:
        pre_layer = pre_layer.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    del cache['i']
    inp_kwargs = cache

    print('Ready.')

    owq_layers = args.meta['owq_layers']
    ratios = args.meta['ratios']
    n_out_dict = {l:0 for l in owq_layers.keys()}
    if args.target_bit is not None:
        n_owq_layers = sum(owq_layers.values())
        
        r = (12 / (16 - args.wbits)) * (args.target_bit - args.wbits)
        # r = (args.target_bit - args.wbits) * 16 / 12
        r /= n_owq_layers

        layer = find_layers(layers[0], layers=[nn.Linear])
        
        for l in owq_layers:
            # for even number of n_out
            n_out = round(layer[l].weight.data.shape[1] * r * ratios[l])
            if n_out % 2 == 1: n_out += 1
            n_out_dict[l] = n_out
    elif args.target_rank is not None:
        for l in owq_layers:
            n_out_dict[l] = args.target_rank
    
    for i in range(len(layers)):
        layer = layers[i].to(dev)
        block_layers = find_layers(layer, layers=[nn.Linear])

        if args.true_sequential:
            sequential = meta['sequential']
        else:
            sequential = [list(block_layers.keys())]
       
        for names in sequential:
            subset = {n: block_layers[n] for n in names}

            gptq_owq = {}
            for name in subset:
                gptq_owq[name] = GPTQ_OWQ(subset[name], n_out=n_out_dict[name])
                gptq_owq[name].quantizer = Quantizer(
                    args.wbits, perchannel=True, sym=args.sym, mse=(args.tuning == 'mse'), group_size=args.groupsize
                )
                gptq_owq[name].quantizer.n_out = n_out_dict[name]
                
            def add_batch(name):
                def tmp(_, inp, out):
                    gptq_owq[name].add_batch(inp[0].data, out.data)
                return tmp
            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(args.nsamples):
                layer(inps[j].unsqueeze(0), **inp_kwargs)
            for h in handles:
                h.remove()
            
            for name in meta['sequential'][0] + meta['sequential'][2]:
                if not args.no_frob_norm:
                    W = subset[name].weight.data.clone().to(torch.float)
                    temp_quantizer = Quantizer(
                        args.wbits, perchannel=True, sym=args.sym, mse=(args.tuning == 'mse'), group_size=args.groupsize
                    )
                    temp_quantizer.find_params(W, weight=True, num=30)
                    W_quant = temp_quantizer.quantize(W)
                    frob_norm_error = (W - W_quant).pow(2).sum(dim=0)
                else:
                    frob_norm_error = None
                out_ids = gptq_owq[name].hessian_sorting(actorder=args.act_order, frob_norm=frob_norm_error)
                gptq_owq[name].quantizer.out_ids = out_ids
                
            # for name in subset: # all
            for name in meta['sequential'][0] + meta['sequential'][2]: # qkv, upgate
                print(f"Quantizing {meta['prefix']}.{i}.{name}")
                
                # before
                # out_ids = gptq_owq[name].quantizer.out_ids
                # sensitivity = gptq_owq[name].H_diag[out_ids]
                # sensitivity_sum[out_ids] += sensitivity / gptq_owq[name].H_diag.mean()
                
                # after
                sensitivity = gptq_owq[name].H_diag
                sensitivity_sum += sensitivity / sensitivity.mean()
                torch.save(out_ids, os.path.join(dirname, f"{meta['prefix']}.{i}.{name}.pth"))
                gptq_owq[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), **inp_kwargs)[0]

        layers[i] = layer.cpu()
        del layer
        del gptq_owq 
        torch.cuda.empty_cache()

        inps, outs = outs, inps
        
    output_path = os.path.join(dirname, f'outidx.pth')
    outidx = sorted(torch.topk(sensitivity_sum, args.target_rank).indices.cpu().tolist())
    print(f"Target rank : {args.target_rank}, nsamples : {args.nsamples}")
    print(f"OutlierIdx : {outidx}")
    print(f"outlieridx is saved to {output_path}")
    torch.save(outidx, output_path)
    model.config.use_cache = use_cache
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'model', type=str,
        help='hugging face model to load'
    )
    parser.add_argument(
        'dataset', type=str,
        help='Where to extract calibration data from. choices = [wikitext2, ptb, c4, custom_path]'
    )
    parser.add_argument(
        '--nsamples', type=int, default=128,
        help='Number of calibration data samples.'
    )
    parser.add_argument(
        '--wbits', type=int, default=16, choices=[2, 3, 4, 16],
        help='The number of bits to use for weight quantization; use 16 for evaluating base model.'
    )
    parser.add_argument(
        '--target_bit', type=float, default=None,
        help='Effctive target bits for OWQ.'
    )
    parser.add_argument(
        '--target_rank', type=int, default=None,
        help='Number of outlier channels for OWQ.(if --target_bit is not given)'
    )
    parser.add_argument(
        '--tuning', type=str, default='mse', choices=['mse', 'minmax'],
        help='Method for quantization parameter tuning.'
    )
    parser.add_argument(
        '--no_frob_norm', action='store_true',
        help='Whether to use Frobenius norm for OWQ.'
    )
    parser.add_argument(
        '--percdamp', type=float, default=.01,
        help='Percent of the average Hessian diagonal to use for dampening.'
    )
    parser.add_argument(
        '--dtype', type=str, default=None,
        help='Data type of model. Use bfloat16 for falcon model family or llama 65B model'
    )
    parser.add_argument(
        '--layers', nargs='+', type=str, default=None,
        help='Layers to apply OWQ.'
    )
    parser.add_argument(
        '--seed', type=int, default=0,
        help='Seed for sampling the calibration data.'
    )
    parser.add_argument(
        '--sym', action='store_true',
        help='Whether to perform symmetric quantization.'
    )
    parser.add_argument(
        '--nearest', action='store_true',
        help='Whether to run the round-to-nearest quantization.'
    ) 
    parser.add_argument(
        '--groupsize', type=int, default=-1,
        help='Groupsize for fine-grained quantization; default uses full row.'
    )
    parser.add_argument(
        '--nearest_owq', action='store_true',
        help='Whether to run the RTN quantization for OWQ.'
    )

    parser.add_argument(
        '--no-eval', action='store_true',
        help='Whether to evaluate model on WikiText-2, PTB and C4'
    )
    parser.add_argument(
        '--save', type=str, default='',
        help='Save quantized checkpoint under this name.'
    )
    parser.add_argument(
        '--load', type=str, default='',
        help='Load fake or 3bit quantized checkpoint.'
    )
    parser.add_argument(
        '--logfile', type=str, default='',
        help='Logging file name'
    )
    parser.add_argument(
        '--fake', action='store_true',
        help='Save fake quantized checkpoint.'
    )
    parser.add_argument(
        '--packing', action='store_true',
        help='Whether to save 3bit quantized model.'
    )
    parser.add_argument(
        '--faster', action='store_true',
        help='Whether to save and load 3bit quantized model using the faster kernel for benchmarking.'
    )
    parser.add_argument(
        '--benchmark', type=int, default=0,
        help='Number of tokens to use for benchmarking.'
    )
    
    parser.add_argument(
        '--act-order', action='store_true',
        help='Whether to apply the activation order GPTQ heuristic'
    )
    parser.add_argument(
        '--true-sequential', action='store_true',
        help='Whether to run in true sequential model.'
    )
    parser.add_argument(
        '--trust_remote_code', action='store_true',
    )
    parser.add_argument(
        '--output_dir', type=str, default='',
        help='outlieridx save dir'
    )
    
    # for out_proj
    parser.add_argument(
        '--perhead', type=int, default=None,
    )
    
    args = parser.parse_args()
    meta = processing_arguments(args)
    args.meta = meta
    device = torch.device('cuda:0')
    
    seed_all(args.seed)
    
    t = 0
    if args.load:
        model = load_model(args.model, args.load, args.faster)
    else:
        model = get_hfmodel(args.model, args.dtype)
    
    if getattr(model.config, 'max_position_embeddings', None):
        args.seqlen = model.config.max_position_embeddings
    elif getattr(model.config, 'max_sequence_length', None):
        args.seqlen = model.config.max_sequence_length
    else:
        args.seqlen = 2048
    
    dataloader = get_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=args.seqlen, train=True
    )
    tick = time.time()
    quantizers = extract_outlieridx(model, dataloader, device, args)
    t = round((time.time() - tick),1)
    print(f"Extract Outlieridx Time : {t}")