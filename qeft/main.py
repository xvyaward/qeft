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
def layerwise_quantize(model, dataloader, dev, args):
    # assert args.no_frob_norm == True
    meta = args.meta
    print('Starting ...')

    use_cache = model.config.use_cache
    layers, pre_layers, post_layers = parsing_layers(model, meta)
    
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
    args.sample = inps[0].clone()
    owq_layers = args.meta['owq_layers']
    ratios = args.meta['ratios']
    n_out_dict = {l:0 for l in owq_layers.keys()}
    if args.target_bit is not None:
        n_owq_layers = sum(owq_layers.values())
        
        r = (12 / (16 - args.wbits)) * (args.target_bit - args.wbits)
        # r = (args.target_bit - args.wbits) * 16 / 12
        r /= n_owq_layers

        layer = find_layers(layers[0], layers=[nn.Linear])
        
        for l, owq in owq_layers.items():
            if owq:
                # for even number of n_out
                n_out = round(layer[l].weight.data.shape[1] * r * ratios[l])
                if n_out % 2 == 1: n_out += 1
                n_out_dict[l] = n_out
    elif args.target_rank is not None:
        for l, owq in owq_layers.items():
            if owq:
                n_out_dict[l] = args.target_rank
    
    quantizers = {}
    for i in tqdm(range(len(layers)),f"Reconstruction Blocks..."):
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
            
            for name in subset:
                if not args.no_frob_norm and (not args.reorder or (args.reorder and name in meta['sequential'][1] + meta['sequential'][3])):
                    W = subset[name].weight.data.clone().to(torch.float)
                    temp_quantizer = Quantizer(
                        args.wbits, perchannel=True, sym=args.sym, mse=(args.tuning == 'mse'), group_size=args.groupsize
                    )
                    temp_quantizer.find_params(W, weight=True, num=40)
                    W_quant = temp_quantizer.quantize(W)
                    frob_norm_error = (W - W_quant).pow(2).sum(dim=0)
                else:
                    frob_norm_error = None
                
                outidx = args.outidx if name not in meta['sequential'][1] + meta['sequential'][3] else None
                out_ids = gptq_owq[name].hessian_sorting(
                    actorder=args.act_order, 
                    frob_norm=frob_norm_error, 
                    outidx=outidx, 
                    )
                gptq_owq[name].quantizer.out_ids = out_ids
                gptq_owq[name].quantizer.n_out = out_ids.numel()
                gptq_owq[name].quantizer.reorder = args.reorder # if name not in meta['sequential'][1] else False

            if not args.no_frob_norm:
                del W
                del W_quant
                del temp_quantizer
                torch.cuda.empty_cache()
            
            for name in subset:
                # print(f"Quantizing {meta['prefix']}.{i}.{name}")
                
                if args.nearest_owq:
                    if gptq_owq[name].quantizer.reorder:
                        gptq_owq[name].fasterquant_nearest_owq_reorder(groupsize=args.groupsize, actorder=args.act_order)
                    else:
                        gptq_owq[name].fasterquant_nearest_owq(groupsize=args.groupsize, actorder=args.act_order)
                else:
                    if gptq_owq[name].quantizer.reorder:
                        gptq_owq[name].fasterquant_reorder(percdamp=args.percdamp, groupsize=args.groupsize, actorder=args.act_order)
                    else:
                        gptq_owq[name].fasterquant(percdamp=args.percdamp, groupsize=args.groupsize, actorder=args.act_order)
                quantizers[f"{meta['prefix']}.{i}.{name}"] = gptq_owq[name].quantizer
                gptq_owq[name].free()
        
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), **inp_kwargs)[0]
            
        for name in list(block_layers.keys()):
            quantizers[f"{meta['prefix']}.{i}.{name}"] = quantizers[f"{meta['prefix']}.{i}.{name}"].cpu()

        layers[i] = layer.cpu()
        del layer
        del gptq_owq 
        torch.cuda.empty_cache()

        inps, outs = outs, inps
            
    if args.reorder:
        args.outidx = args.outidx.cpu()
        make_reorder(model, quantizers, args)
    model.config.use_cache = use_cache
    
        
    return quantizers

@torch.no_grad()
def eval_ppl(model, testenc, dev, args):
    meta = args.meta
    print('Evaluating ...')

    testenc = testenc.input_ids
    # nsamples = 8
    nsamples = testenc.numel() // args.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers, pre_layers, post_layers = parsing_layers(model, meta)

    for pre_layer in pre_layers:
        pre_layer = pre_layer.to(dev)
    
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, args.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )

    cache = {kw:None for kw in meta['inp_kwargs']}
    cache['i'] = 0

    
    if args.reorder:
        def dynamic_reorder_hook(module, inp):
            return torch.index_select(inp[0], -1, module.reorder_ids.to(inp[0].device))
        
        reorder_handles = []
        for layer in layers:
            reorder_handles.append(layer.self_attn.o_proj.register_forward_pre_hook(dynamic_reorder_hook))
    
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
    for i in range(nsamples):
        batch = testenc[:, (i * args.seqlen):((i + 1) * args.seqlen)].to(dev)
        # if 'gemma' in args.model:
        #     batch[:, 0] = model.config.bos_token_id
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    
    for pre_layer in pre_layers:
        pre_layer = pre_layer.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    del cache['i']
    inp_kwargs = cache
    
    for i in tqdm(range(len(layers))):
        layer = layers[i].to(dev)
        
        if args.nearest:
            subset = find_layers(layer, layers=args.meta['linears'])
            for name in subset:
                quantizer = Quantizer(args.wbits, perchannel=True, sym=args.sym, mse=False)
                W = subset[name].weight.data
                quantizer.find_params(W, weight=True)
                subset[name].weight.data = quantizer.quantize(W).to(next(iter(layer.parameters())).dtype)

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), **inp_kwargs)[0]
        # import code; code.interact('after first layer', local=dict(globals(), **locals()))

        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    for post_layer in post_layers:
        post_layer = post_layer.to(dev)
    
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if post_layer in post_layers:
            hidden_states = post_layer(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[
            :, (i * args.seqlen):((i + 1) * args.seqlen)
        ][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * args.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * args.seqlen))
    print(ppl.item())

    model.config.use_cache = use_cache
    return ppl.item()

def benchmark(model, input_ids, args):
    meta = args.meta
    layers, _, _ = parsing_layers(model, meta)
    
    dev = model.gpus[0] if hasattr(model, 'gpus') else model.device
    input_ids = input_ids.to(dev)
    torch.cuda.synchronize()

    cache = {'past': None}
    def clear_past(i): # for memory collect
        def tmp(layer, inp, out):
            if cache['past']:
                cache['past'][i] = None
        return tmp
    
    for i, layer in enumerate(layers):
        layer.register_forward_hook(clear_past(i))

    print('Benchmarking ...')

    loss = nn.CrossEntropyLoss()
    tot = 0.
    torch.cuda.empty_cache()
    def sync():
        if hasattr(model, 'gpus'):
            for gpu in model.gpus:
                torch.cuda.synchronize(gpu)
        else:
            torch.cuda.synchronize()

    with torch.no_grad():
        times = []
        if args.ft:
            start_pos = None
            for i in range(input_ids.numel()):
                tick = time.perf_counter()
                out = model(input_ids[:, i].reshape(1,-1), start_pos=start_pos)

                start_pos = out.start_pos
                sync()
                t = time.perf_counter() - tick
                times.append(t)
                if i != input_ids.numel() - 1:
                    tot += loss(out.logits[0].to(dev), input_ids[:, (i + 1)].to(dev)).float()
                # print(i, t)
        else:
            for i in range(input_ids.numel()):
                tick = time.perf_counter()
                out = model(input_ids[:, i].reshape(1,-1),
                            past_key_values=cache['past'])
                sync()
                t = time.perf_counter() - tick
                times.append(t)
                if i != input_ids.numel() - 1:
                    tot += loss(out.logits[0].to(dev), input_ids[:, (i + 1)].to(dev)).float()
                # print(i, t)
                cache['past'] = list(out.past_key_values)
        sync()
        
        print(f'Median(second): {np.median(times)}')
        print(f'Min(second): {np.min(times)}')
        print(f'PPL:', torch.exp(tot / (input_ids.numel() - 1)).item())
    
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
        '--trained_checkpoint', type=str, default=None,
        help='Load trained checkpoint.'
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
        '--training', action='store_true',
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
        '--reorder', action='store_true',
        help='Whether to reorder matrix.'
    )
    parser.add_argument(
        '--outidx_file', type=str, default=None
    )
    
    parser.add_argument(
        '--ft', action='store_true',
    )
    
    args = parser.parse_args()
    meta = processing_arguments(args)
    args.meta = meta
    device = torch.device('cuda:0')
    
    seed_all(args.seed)
    
    t = 0
    if args.load:
        model = load_model(model_name_or_path=args.model, checkpoint_path=args.load, training=args.training, attn_implementation='ft' if args.ft else None)
    else:
        model = get_hfmodel(model_name_or_path=args.model, dtype=args.dtype, attn_implementation='ft' if args.ft else None)
    # import code; code.interact('model inspect', local=dict(globals(), **locals()))
    
    if getattr(model.config, 'max_position_embeddings', None):
        args.seqlen = model.config.max_position_embeddings
    elif getattr(model.config, 'max_sequence_length', None):
        args.seqlen = model.config.max_sequence_length
    else:
        args.seqlen = 2048
    # args.seqlen = 2048
    
    if args.outidx_file:
        args.outidx = torch.tensor(torch.load(args.outidx_file, map_location=device)).sort()[0]
    else:
        args.outidx = None

    if not args.load and args.wbits < 16 and not args.nearest:
        dataloader = get_loaders(
            args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=args.seqlen, train=True
        )
        tick = time.time()
        quantizers = layerwise_quantize(model, dataloader, device, args)
        t = round((time.time() - tick),1)
        print(f"Running Time : {t}")
    
    # benchmark
    if args.benchmark:
        dataloader = get_loaders(
            args.dataset, nsamples=1, seed=args.seed, model=args.model, seqlen=args.seqlen, train=True
        )
        gpus = [torch.device('cuda:%d' % i) for i in range(torch.cuda.device_count())]
        if len(gpus) > 1:
            model_multigpu(model, gpus, args.meta)
        else:
            model = model.to(device)
        
        if isinstance(dataloader,list):
            input_ids = dataloader[0][0][:,:args.benchmark]
        else:
            input_ids = dataloader.input_ids[:, :args.benchmark]
        benchmark(model, input_ids, args)
        exit()

    # eval
    t1 = time.time()
    ppl_scores = []
    if not args.no_eval:
        ppl_tasks = ['wikitext2','ptb', 'c4']
        for dataset in ppl_tasks:
            testloader = get_loaders(
                dataset, seed=args.seed, model=args.model, seqlen=args.seqlen, train=False
            )
            print(dataset)
            ppl_score = eval_ppl(model, testloader, device, args)
            ppl_scores.append((dataset,ppl_score))
            break
    t2 = time.time() - t1
    
    # logging
    if args.logfile:
        dtype = model.dtype
        with open(f'{args.logfile}','a') as fp:
            add_str = f"\nlayers : {args.layers}" + f"| target_bit : {args.target_bit}\n" if args.target_bit is not None else '\n'
            fp.write(f"model : {args.model} | owq time : {round(t/60,1)}m / eval time : {round(t2/60,1)}m | seed : {args.seed} {f'| dtype : {dtype}'}{add_str}")
            for i in range(len(ppl_scores)):
                fp.write(f"{ppl_scores[i][1]} ")
            fp.write(f"\n\n")
    # save
    if args.save:
        save_model(model, quantizers, args.save, args.packing, args.fake)