import math
import torch
import torch.nn as nn
from typing import Union,Optional
from transformers import AutoModelForCausalLM

from collections import OrderedDict
import os
import json
import re

from qeft.quant import lm_pack, make_quant
from qeft.qlinear import QuantLinear
from qeft.utils.misc import find_layers, interpret_dtype, parsing_layers, get_model_config
from qeft.monkeypatch.ftllama_modeling import convert_model_to_ft

# for swa
# from axolotl.monkeypatch.qspec_utils import replace_forward_swa_flash_attention_2, replace_update_causal_mask
# from transformers.utils import is_flash_attn_2_available, is_flash_attn_greater_or_equal_2_10

def model_multigpu(model, gpus, meta=None, model_name=None):
    assert meta is not None or model_name is not None, "at least one of 'meta' or 'model_name' argument must not None"
    
    if meta is None:
        meta = get_model_config(model_name)

    layers, pre_layers, post_layers = parsing_layers(model=model, meta=meta)
    
    for pre_layer in pre_layers:
        pre_layer = pre_layer.to(gpus[0])
    
    for post_layer in post_layers:
        post_layer = post_layer.to(gpus[0])
    
    model.lm_head = model.lm_head.to(gpus[0])

    class MoveModule(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.dev = next(iter(self.module.parameters())).device
        def forward(self, *inp, **kwargs):
            inp = list(inp)
            if inp[0].device != self.dev:
                inp[0] = inp[0].to(self.dev)
            for key in kwargs:
                if hasattr(kwargs[key], 'device') and kwargs[key].device != self.dev:
                    kwargs[key] = kwargs[key].to(self.dev)
            tmp = self.module(*inp, **kwargs)
            return tmp

    pergpu = math.ceil(len(layers) / len(gpus))
    for i in range(len(layers) - 1):
        layers[i] = MoveModule(layers[i].to(gpus[i // pergpu]))
    layers[-1] = MoveModule(layers[-1].to(gpus[0]))

    model.gpus = gpus

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x
    
def get_hfmodel(model_name_or_path: str,
                dtype='auto',
                device_map='cpu',
                trust_remote_code=False,
                **kwargs
                ):
    
    # for fast model loading
    org_kaiming_uniform = torch.nn.init.kaiming_uniform_
    org_uniform = torch.nn.init.uniform_
    org_normal = torch.nn.init.normal_
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    
    
    ft = False
    if kwargs.get('attn_implementation') == 'ft':
        assert 'llama' in model_name_or_path.lower() or 'vicuna' in model_name_or_path.lower()
        kwargs.pop('attn_implementation')
        ft = True
        
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, 
        torch_dtype=dtype,
        device_map=device_map, 
        trust_remote_code=trust_remote_code,
        **kwargs
    )
    
    if ft:
        convert_model_to_ft(model)

    torch.nn.init.kaiming_uniform_ = org_kaiming_uniform
    torch.nn.init.uniform_ = org_uniform
    torch.nn.init.normal_ = org_normal
    
    return model

def load_model(model_name_or_path,
                checkpoint_path,
                training: Optional[bool] = False,
                dtype = None,
                device: Optional[Union[int, str, torch.device]] = 'cuda:0',
                cpu_load: Optional[bool] = True,
                **kwargs
                ):
    if 'base_path' in torch.load(checkpoint_path).keys(): # wct
        return load_wctmodel(model_name_or_path, checkpoint_path, training, dtype, device, cpu_load, **kwargs)
    else:
        return load_owqmodel(model_name_or_path, checkpoint_path, training, dtype, device, cpu_load, **kwargs)

def hfmodel_to_owqmodel(model, ckpt, training: Optional[bool] = False,
                        device: Optional[Union[int, str, torch.device]] = 'cuda:0'):
    
    print(f"Loading model ....")
    
    if ckpt['packing']:
        make_quant(model, ckpt['quantinfos'])
        # import code; code.interact('hfmodel_to_owqmodel', local=dict(globals(), **locals()))
                
        model.load_state_dict(ckpt['model_state_dict'], strict=False)

        qlayers = find_layers(model, [QuantLinear])
        for name in qlayers:
            qlayers[name].set_kernel(training)
    else:
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
            
    if model.device == 'cpu' and device not in ['auto', 'cpu']:
        model = model.to(device)
    
    del ckpt
    import gc; gc.collect()
    torch.cuda.empty_cache()
    
    print("Done.")
    return model

def load_owqmodel(model_name_or_path,
                  checkpoint_path,
                  training: Optional[bool] = False,
                  dtype = None,
                  device: Optional[Union[int, str, torch.device]] = 'cuda:0',
                  cpu_load: Optional[bool] = True,
                  **kwargs,
                  ):
    if not isinstance(device, torch.device) and device not in ['auto', 'cpu']:
        device = torch.device(device)
    device_map = 'cpu' if cpu_load else device
    ckpt = torch.load(checkpoint_path)

    if dtype == None:
        dtype = ckpt['dtype']
    else:
        dtype = interpret_dtype(dtype)
    try:
        import accelerate
        
        with accelerate.init_empty_weights():
            model = get_hfmodel(model_name_or_path,
                                dtype=dtype,
                                device_map=device_map,
                                **kwargs)
    except:
        model = get_hfmodel(model_name_or_path,
                            dtype=dtype,
                            device_map=device_map,
                            **kwargs)
        
    model = hfmodel_to_owqmodel(model, ckpt, training, device)
    
    ## added
    model = model.to(device=device)

    return model

def replace_oweight(model, ckpt_wct):
    oweight_state_dict = ckpt_wct['oweight_state_dict']
    
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            key = name.replace('.oweight', '')
            if key in oweight_state_dict:
                module.oweight.data = oweight_state_dict[key].data.to(module.oweight.dtype)
            
    del ckpt_wct
    import gc; gc.collect()
    torch.cuda.empty_cache()
        
    return model

def load_wctmodel(model_name_or_path,
                  checkpoint_path,
                  training: Optional[bool] = True,
                  dtype=None,
                  device: Optional[Union[int, str, torch.device]] = 'cuda:0',
                  cpu_load: Optional[bool] = True,
                  **kwargs,
                  ):
    
    if os.path.isdir(checkpoint_path):
        ckpt = torch.load(os.path.join(checkpoint_path, 'model.pth'))
    else:
        ckpt = torch.load(checkpoint_path)
        
    model = load_owqmodel(model_name_or_path, ckpt['base_path'], training, dtype, device, cpu_load, **kwargs)
    replace_oweight(model, ckpt)
        
    return model

def save_model(model, 
               quantizers,
               save_path,
               packing:bool,
               fake:bool):
    
    dtype = model.dtype
    wbits = list(quantizers.values())[0].bits
    group_size = list(quantizers.values())[0].group_size
    
    dirname = os.path.dirname(save_path)
    if not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)
        
    if fake:
        ckpt_path = save_path.replace('.pt', '_fake.pt')
        model_state_dict = model.state_dict()
        out_ids_dict = {name : quantizers[name].out_ids for name in quantizers}
        
        torch.save({
            'model_state_dict': model_state_dict,
            'out_ids_dict': out_ids_dict,
            'packing': False,
            'dtype' : dtype,
            'bits' : wbits,
            'group_size' : group_size,
            }, ckpt_path)

        print(f"fake quantized model is saved to {ckpt_path}")
    if packing:
        assert wbits in [3, 4], f"{wbits}bits is not supported."
        lm_pack(model, quantizers)
        model_state_dict = model.state_dict()
        from argparse import Namespace
        quantinfos = {n: Namespace(**{'bits':quantizers[n].bits, 
                          'sym':getattr(quantizers[n],'sym',False),
                          'group_size':getattr(quantizers[n],'group_size',-1), 
                          'n_out':getattr(quantizers[n],'n_out',0), 
                          'reorder':getattr(quantizers[n], 'reorder', False),
                          }) for n in quantizers}
        
        torch.save({
            'model_state_dict': model_state_dict,
            'quantinfos': quantinfos,
            'packing': True,
            'dtype' : dtype,
            'bits' : wbits, 
            'group_size' : group_size,
            }, save_path)
        print(f"{wbits}bit quantized packing model is saved to {save_path}")   
    
def save_wctmodel(model,
                  base_path,
                  output_dir):
    model_state_dict = OrderedDict()
    for name, param in model.named_parameters():
        if 'oweight' in name:
            model_state_dict[name.replace('.oweight', '')] = param.data.to(model.dtype)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'model.pth')
    torch.save({'oweight_state_dict': model_state_dict,
                'base_path' : os.path.abspath(base_path)}, save_path)
    
    print(f"fine-tuned model is saved to {save_path}.")
