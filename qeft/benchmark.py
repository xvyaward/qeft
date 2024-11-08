# Usage:
# Please first install awq/kernels
# then directly run CUDA_VISIBLE_DEVICES=0 python benchmark.py
import argparse
import torch
import time
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, modeling_utils, BitsAndBytesConfig, AwqConfig

from qeft.quant import *
from qeft.utils.misc import find_layers
from qeft.utils.modelutils import get_hfmodel, load_model, model_multigpu

from peft import (
    LoraConfig,
    get_peft_model
)

def device_warmup(device: str):
    warm_up = torch.randn((4096, 4096)).to(device)
    for i in range(100):
        torch.mm(warm_up, warm_up)
        
def skip(*args, **kwargs):
    pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type", type=str, default="LLaMa", help="type of the model"
    )
    parser.add_argument(
        "--ft",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--old",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/data/llm/checkpoints/vicuna-hf/vicuna-7b",
        help="path to the model",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="",
        help="path to the model",
    )
    parser.add_argument(
        "--load",
        type=str,
        default="",
        help="path to the qeft model",
    )
    parser.add_argument(
        "--qalora",
        type=str,
        default="",
        help="path to the autogptq model",
    )
    parser.add_argument(
        "--v1",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--v2",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="",
        help="path to the autogptq model",
    )
    parser.add_argument(
        "--lora",
        default=False,
        action="store_true",
    )
    parser.add_argument("--q_group_size", type=int, default=128)
    parser.add_argument(
        "--verbose",
        default=False,
        action="store_true",
        help="Wheter to print more information.",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=2048,
        help="maximum sequence length for kv cache",
    )
    parser.add_argument(
        "--max_batch_size", type=int, default=1, help="maximum batch size for kv cache"
    )
    args = parser.parse_args()

    # tinychat.utils.constants.max_batch_size = args.max_batch_size
    # tinychat.utils.constants.max_seq_len = args.max_seq_len
    # from tinychat.models import FalconForCausalLM, LlamaForCausalLM, MPTForCausalLM
    # from transformers.models.llama.modeling_llama_4_41_2_ft import LlamaForCausalLM

    modeling_utils._init_weights = False
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.kaiming_normal_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip

    device = "cuda:0"
    # exLLaMA benchmarking parameters.
    context_length = 64
    gen_length = 256
    input_ids = [1 for _ in range(context_length)]

    add_str = ''

    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    assert args.model_type.lower() in [
        "llama",
        "falcon",
        "mpt",
    ], "We only support llama & falcon & mpt now"
    
    attn_implementation = 'ft' if args.ft else None
    # fast_model = model_type_dict[args.model_type.lower()](config).half()
    
    if args.method == 'fp':
        if '65b' in args.model_path.lower() or '70b' in args.model_path.lower(): # use 2-gpu
            model = get_hfmodel(args.model_path, device_map='cpu', attn_implementation=attn_implementation)
        else:
            model = get_hfmodel(model_name_or_path=args.model_path, device_map=device, dtype=torch.float16, attn_implementation=attn_implementation)
    elif args.method == 'platypus':
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            load_in_8bit=True,
            torch_dtype=torch.float16,
            device_map=device,
            attn_implementation=attn_implementation
        )
        config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=['gate_proj', 'down_proj', 'up_proj'],
            bias="none",
            task_type="CAUSAL_LM")

        model = get_peft_model(model, config)
    elif args.method == 'qlora':
        import bitsandbytes as bnb
        
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            load_in_4bit=True,
            device_map=device,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4',
            ),
            torch_dtype=torch.float16,
            attn_implementation=attn_implementation
        )
        
        cls = bnb.nn.Linear4bit
        lora_module_names = set()
        for name, module in model.named_modules():
            if isinstance(module, cls):
                names = name.split('.')
                lora_module_names.add(names[0] if len(names) == 1 else names[-1])
        if 'lm_head' in lora_module_names:
            lora_module_names.remove('lm_head')
        target_modules = list(lora_module_names)
        config = LoraConfig(
            r=64,
            lora_alpha=16,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM")
        model = get_peft_model(model, config)
    elif args.method == 'qalora':
        if args.v1:
            disable_exllama=False
            disable_exllamav2=True
            add_str = '| v1'
        elif args.v2:
            disable_exllama=False
            disable_exllamav2=False
            add_str = '| v2'
        else:
            disable_exllama=True
            disable_exllamav2=True
            add_str = '| old'
        
        if not args.ft:
            add_str += '_hf'
        
        from auto_gptq import AutoGPTQForCausalLM
        
        model = AutoGPTQForCausalLM.from_quantized(args.qalora, device=device,
                                                   disable_exllama=disable_exllama,
                                                   disable_exllamav2=disable_exllamav2,
                                                   )
    elif args.method == 'qeft':
        model = load_model(args.model_path, args.load, device=device, attn_implementation=attn_implementation)

    elif args.method == 'awq':
        model = get_hfmodel(args.model_path, attn_implementation=attn_implementation)
        layers = find_layers(model)
        
        if args.lora:
            cls = nn.Linear
            lora_module_names = set()
            for name, module in model.named_modules():
                if isinstance(module, cls):
                    names = name.split('.')
                    lora_module_names.add(names[0] if len(names) == 1 else names[-1])
            if 'lm_head' in lora_module_names:
                lora_module_names.remove('lm_head')                
            target_modules = list(lora_module_names)
            config = LoraConfig(
                r=64,
                lora_alpha=16,
                target_modules=target_modules,
                bias="none",
                task_type="CAUSAL_LM")
            model = get_peft_model(model, config)
        
            # for validate
            layers = find_layers(model)
        
            for name in ["lm_head"]:
                if name in layers:
                    del layers[name]
            real_layers = {}
            for k, v in layers.items():
                if k.endswith('base_layer'):
                    real_layers[k] = v
            layers = real_layers
        
        # make_quant_linear(model, layers, w_bit=4, groupsize=128, device=device)
        del layers
    else:
        raise ValueError
    # import code; code.interact('model check', local=dict(globals(), **locals()))

    model.eval()
    if '65b' in args.model_path.lower() or '70b' in args.model_path.lower(): # use 2-gpu
        model_multigpu(model, [torch.device('cuda:0'), torch.device('cuda:1')], model_name=args.model_path)
    else:    
        model = model.to(device)
    # # dynamic reordering
    # def sparse_to_dense_ids(sparse_ids, length):
    #     assert len(sparse_ids) < length
        
    #     temp_mask = torch.full([length], True, device=sparse_ids.device)
    #     temp_mask[sparse_ids] = False
    #     dense_ids = torch.cat([torch.arange(length, device=sparse_ids.device)[temp_mask], sparse_ids])
    #     return dense_ids
    # def ln_reorder_hook(module, inp, out):
    #     return torch.index_select(out, -1, module.ids)
    
    # handles = []
    # for block in model.model.layers:
    #     out_ids = block.self_attn.k_proj.outlieridx
    #     ln_attn = block.input_layernorm
    #     ln_attn.ids = sparse_to_dense_ids(torch.randperm(ln_attn.weight.shape[0]).to(out_ids.device)[:out_ids.shape[0]],ln_attn.weight.shape[0])
    #     handles.append(ln_attn.register_forward_hook(ln_reorder_hook))
    #     ln_ffn = block.post_attention_layernorm
    #     ln_ffn.ids = sparse_to_dense_ids(torch.randperm(ln_attn.weight.shape[0]).to(out_ids.device)[:out_ids.shape[0]],ln_attn.weight.shape[0])
    #     handles.append(ln_ffn.register_forward_hook(ln_reorder_hook))

    # tune_all_wqlinears(model)
    # make_quant_attn(model, device)
    # make_quant_norm(model)
    # make_fused_mlp(model)
    device_warmup(device)

    print("huggingface ckpt loaded")
    # print(model)
    
    time_lis = []
    torch.cuda.reset_peak_memory_stats()
    print("Benchmarking...")
    with torch.inference_mode():
        if args.ft:
            start_pos = 0
            for i in range(gen_length):
                torch.cuda.synchronize()

                if i == 0:
                    inputs = torch.as_tensor([input_ids], device=device)
                else:
                    inputs = torch.as_tensor([[token]], device=device)
                t_st = time.time()
                out = model(inputs, start_pos=start_pos, use_cache=False)
                torch.cuda.synchronize()
                t_ed = time.time()
                start_pos += out.logits.shape[1]

                time_lis.append(t_ed - t_st)
                
                token = out.logits[:, -1].max(1)[1].unsqueeze(1)
                if args.verbose:
                    print(i, token, np.median(time_lis))
        else:
            last_key_values = None
            for i in range(gen_length):
                torch.cuda.synchronize()

                if i == 0:
                    inputs = torch.as_tensor([input_ids], device=device)
                else:
                    inputs = torch.as_tensor([[token]], device=device)
                t_st = time.time()
                out = model(inputs, past_key_values=last_key_values)
                torch.cuda.synchronize()
                t_ed = time.time()
                out, last_key_values = out.logits, out.past_key_values

                time_lis.append(t_ed - t_st)
                
                token = out[:, -1].max(1)[1].unsqueeze(1)
                if args.verbose:
                    print(i, token, np.median(time_lis))
    if args.output_file:
        with open(args.output_file, 'a') as f:
            f.write(f"Method : {args.method} | {args.owq} {add_str} Speed: {1 / np.median(time_lis):.2f} tokens per second. ({np.median(time_lis) * 1000:.2f}ms per token)\n")
    else:
        print(f"Max memory usage : {torch.cuda.max_memory_reserved() / 1024 / 1024}MB")
        print(f"Speed: {1 / np.median(time_lis):.2f} tokens per second. ({np.median(time_lis) * 1000:.2f}ms per token)")


if __name__ == "__main__":
    main()
