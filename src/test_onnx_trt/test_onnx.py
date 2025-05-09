import os
import nvtx

import torch
import torch.nn as nn
import torch.nn.functional as F

import time

import nvtx

from typing import Any, Dict, List, Optional, Tuple, Union

import tensorrt as trt
from tensorrt_llm._utils import torch_dtype_to_str, to_json_file
from tensorrt_llm.builder import Builder
from tensorrt_llm.logger import logger
from tensorrt_llm.runtime.session import Session, TensorInfo

from tensorrt_llm._utils import (mpi_rank, str_dtype_to_torch, str_dtype_to_trt,
                        supports_inflight_batching, torch_dtype_to_trt,
                        trt_dtype_to_torch)

from flash_attn.layers.rotary import apply_rotary_emb
from flash_attn import flash_attn_varlen_func


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)   
    
def apply_rotary_pos_emb_vision(tensor: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    orig_dtype = tensor.dtype
    tensor = tensor.float()
    cos = freqs.cos()
    sin = freqs.sin()
    cos = cos.unsqueeze(1).repeat(1, 1, 2).unsqueeze(0).float()
    sin = sin.unsqueeze(1).repeat(1, 1, 2).unsqueeze(0).float()
    output = (tensor * cos) + (rotate_half(tensor) * sin)
    output = output.to(orig_dtype)
    return output
    
def apply_rotary_pos_emb_flashatt(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos = cos.chunk(2, dim=-1)[0].contiguous()
    sin = sin.chunk(2, dim=-1)[0].contiguous()
    q_embed = apply_rotary_emb(q.float(), cos.float(), sin.float()).type_as(q)
    k_embed = apply_rotary_emb(k.float(), cos.float(), sin.float()).type_as(k)
    return q_embed, k_embed
    
class MyFA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, len_q, len_k) -> torch.Tensor:
        # ctx.save_for_backward(input)
        # return flash_attn_varlen_func(q, k, v, len_q, len_k, 64, 64)
        
        return q + k

    @staticmethod
    def symbolic(g: torch.Graph, q, k, v, len_q, len_k) -> torch.Value:
        return g.op("MyFA", q, k, v, len_q, len_k)
    
_myfa = MyFA.apply

class MyModel(nn.Module):
    
    def __init__(self, dim: int = 1280, num_heads: int = 16) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        
        self.cu_seqlens = torch.load('cu_seqlens.pt').to('cuda')
        self.cu_window_seqlens = torch.load('cu_window_seqlens.pt').to('cuda')
            
    ## eager attn
    def forward_navie(self,
                hidden_states: torch.Tensor,
                attention_mask: torch.Tensor,
                rotary_pos_emb: torch.Tensor = None) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        
        q = apply_rotary_pos_emb_vision(q.unsqueeze(0),
                                        rotary_pos_emb).squeeze(0)
        k = apply_rotary_pos_emb_vision(k.unsqueeze(0),
                                        rotary_pos_emb).squeeze(0)
        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)

        attn_weight = torch.matmul(q, k.transpose(-1, -2)) / torch.sqrt(torch.tensor(self.head_dim).to(q.device))
        attn_weight = attn_weight + attention_mask
        attn_weight = nn.functional.softmax(attn_weight, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weight, v)
        attn_output = attn_output.transpose(0, 1)
        attn_output = attn_output.reshape(seq_length, -1)
        
        attn_output = self.proj(attn_output)
        return attn_output
    
    def forward_sdpa(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        if position_embeddings is None:
            
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            cos = emb.cos().float()
            sin = emb.sin().float()
        else:
            cos, sin = position_embeddings
        # q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)
        q = apply_rotary_pos_emb_vision(q.unsqueeze(0),
                                        rotary_pos_emb).squeeze(0)
        k = apply_rotary_pos_emb_vision(k.unsqueeze(0),
                                        rotary_pos_emb).squeeze(0)

        # attention_mask = torch.zeros([1, seq_length, seq_length], device=q.device, dtype=torch.bool)
        # for i in range(1, len(cu_seqlens)):
        #     attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = True
        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        attention_mask = attention_mask.to(q.dtype)
        attn_output = F.scaled_dot_product_attention(q, k, v, attention_mask, dropout_p=0.0)
        attn_output = attn_output.transpose(0, 1)
        attn_output = attn_output.reshape(seq_length, -1)
        attn_output = self.proj(attn_output)
        return attn_output

    ## flash attn
    def forward_flash(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
            
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
            
        q, k = apply_rotary_pos_emb_flashatt(q.unsqueeze(0), k.unsqueeze(0), cos, sin)
        # q = apply_rotary_pos_emb_vision(q.unsqueeze(0),
        #                                 rotary_pos_emb).squeeze(0)
        # k = apply_rotary_pos_emb_vision(k.unsqueeze(0),
        #                                 rotary_pos_emb).squeeze(0)
        
        q = q.squeeze(0)
        k = k.squeeze(0)

        if attention_mask[0][0][-1] < 0:
            # window atten
            cu_seqlens = self.cu_window_seqlens
        else:
            cu_seqlens = self.cu_seqlens
        # cu_seqlens = self.cu_window_seqlens
        
        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        
        dtype = q.dtype
        q = q.to(torch.bfloat16)
        k = k.to(torch.bfloat16)
        v = v.to(torch.bfloat16)
        attn_output = flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen).reshape(
            seq_length, -1
        )
        
        # attn_output = _myfa(q, k, v, cu_seqlens, cu_seqlens).reshape(seq_length, -1)
        attn_output = attn_output.to(dtype)
        
        attn_output = self.proj(attn_output)
        # attn_output = self.proj(hidden_states)
        return attn_output


class ModelWrapper(nn.Module):
    def __init__(self, dim: int = 1280, num_heads: int = 16):
        super().__init__()
        self.layers = 10
        self.model_seq = nn.Sequential(*[MyModel(dim, num_heads) for i in range(self.layers)])

    def forward_navie(self, hidden_states, full_attention_mask, rotary_pos_emb):
        for i in range(self.layers):
            hidden_states = self.model_seq[i].forward_navie(hidden_states, full_attention_mask, rotary_pos_emb)
        return hidden_states
    def forward_sdpa(self, hidden_states, full_attention_mask, rotary_pos_emb):
        for i in range(self.layers):
            hidden_states = self.model_seq[i].forward_sdpa(hidden_states, full_attention_mask, rotary_pos_emb)
        return hidden_states
    def forward_flash(self, hidden_states, full_attention_mask, rotary_pos_emb):
        for i in range(self.layers):
            hidden_states = self.model_seq[i].forward_flash(hidden_states, full_attention_mask, rotary_pos_emb)
        return hidden_states

def torch_forward(model: nn.Module, hidden_states, full_attention_mask, rotary_pos_emb):
    for i in range(3):
        # stream = torch.cuda.Stream(torch.cuda.current_device())
        # torch.cuda.set_stream(stream)
        start = time.time()
        with nvtx.annotate('TORCH FLASH ATTN', color='yellow'):
            res = model.forward_flash(hidden_states, full_attention_mask, rotary_pos_emb)
        # stream.synchronize()
        print(f'PyTorch duration {time.time() - start} s')
    return res

def onnx_forward(hidden_states, full_attention_mask, rotary_pos_emb):
    import onnxruntime as ort
    import numpy as np
    import onnx
    from onnx import helper,shape_inference
    

    model_path = "custom_model.onnx"
    session = ort.InferenceSession(model_path)

    input_names = [input.name for input in session.get_inputs()]
    output_names = [output.name for output in session.get_outputs()]

    
    hidden_states = hidden_states.detach().cpu().numpy()
    full_attention_mask = full_attention_mask.detach().cpu().numpy()
    rotary_pos_emb = rotary_pos_emb.detach().cpu().numpy()

    inputs = {input_names[0]: hidden_states, input_names[1]: full_attention_mask, input_names[2]: rotary_pos_emb}
    # inputs = {input_names[0]: hidden_states}
    # inputs = {}

    outputs = session.run(output_names, inputs)

    output_data = outputs[0]
    return output_data

def trt_forward(x: list, dtype: str='float32'):
    # engine_file = 'model_FP16.plan'
    engine_file = 'custom_model.engine'
    with open(engine_file, 'rb') as f:
        engine_buffer = f.read()

    visual_encoder_session = Session.from_serialized_engine(
        engine_buffer)
    
    dummy_input = x
    visual_features = {
        'input': dummy_input[0],
        'full_attention_mask': dummy_input[1].to(str_dtype_to_torch(dtype)),
        'rotary_pos_emb': dummy_input[2].to(str_dtype_to_torch(dtype)),
    }
    
    tensor_info = [
        TensorInfo('input', str_dtype_to_trt('float32'), dummy_input[0].shape),
        TensorInfo('full_attention_mask', str_dtype_to_trt(dtype), dummy_input[1].shape),
        TensorInfo('rotary_pos_emb', str_dtype_to_trt(dtype), dummy_input[2].shape),
    ]
    visual_output_info = visual_encoder_session.infer_shapes(
        tensor_info)
    visual_encoder_session.set_shapes(visual_features)
    visual_outputs = {
        t.name:
        torch.empty(tuple(t.shape),
                    dtype=trt_dtype_to_torch(t.dtype),
                    device=dummy_input[0].device)
        for t in visual_output_info
    }

    for i in range(3):
        stream = torch.cuda.Stream(torch.cuda.current_device())
        torch.cuda.set_stream(stream)
        start = time.time()
        with nvtx.annotate('TRT ATTN', color='red'):
            ok = visual_encoder_session.run(visual_features, visual_outputs, stream.cuda_stream)
            assert ok, "Runtime execution failed for vision encoder session"
            stream.synchronize()
        total_trt_duration = time.time() - start
        print(f'TRT duration {total_trt_duration} s')
                
    image_embeds = visual_outputs['output']
    return image_embeds

def trt_forward_qwen(dtype: str='float32'):
    engine_file = 'model_FP16.plan'
    with open(engine_file, 'rb') as f:
        engine_buffer = f.read()

    visual_encoder_session = Session.from_serialized_engine(
        engine_buffer)
    
    hidden_states = torch.load('hidden_states.pt')
    rotary_pos_emb = torch.load('rotary_pos_emb.pt')
    full_attention_mask = torch.load('full_attention_mask.pt')
    window_attention_mask = torch.load('window_attention_mask.pt')
    
    visual_features = {
        'input': hidden_states.to(str_dtype_to_torch(dtype)),
        'rotary_pos_emb': rotary_pos_emb.to(str_dtype_to_torch(dtype)),
        'full_attention_mask': full_attention_mask.to(str_dtype_to_torch(dtype)),
        'window_attention_mask': window_attention_mask.to(str_dtype_to_torch(dtype)),
    }
    
    tensor_info = [
        TensorInfo('input', str_dtype_to_trt(dtype), hidden_states.shape),
        TensorInfo('rotary_pos_emb', str_dtype_to_trt(dtype), rotary_pos_emb.shape),
        TensorInfo('full_attention_mask', str_dtype_to_trt(dtype), full_attention_mask.shape),
        TensorInfo('window_attention_mask', str_dtype_to_trt(dtype), window_attention_mask.shape),
    ]
    visual_output_info = visual_encoder_session.infer_shapes(
        tensor_info)
    visual_encoder_session.set_shapes(visual_features)
    visual_outputs = {
        t.name:
        torch.empty(tuple(t.shape),
                    dtype=trt_dtype_to_torch(t.dtype),
                    device=hidden_states.device)
        for t in visual_output_info
    }
    stream = torch.cuda.Stream(torch.cuda.current_device())
    torch.cuda.set_stream(stream)
    ok = visual_encoder_session.run(visual_features, visual_outputs, stream.cuda_stream)
    assert ok, "Runtime execution failed for vision encoder session"
    image_embeds = visual_outputs['output']
    return image_embeds

def generate_trt(dtype: str='float32'):
    logger=trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    profile = builder.create_optimization_profile()

    # onnx_file = f'tmp/trt_engines/Qwen2_5-VL/fp16/1-gpu/vision/onnx/model.onnx'
    onnx_file = f'custom_model.onnx'
    engine_file = f'custom_model.engine'
    config_file = f'custom_model.config'
    logger.log(trt.Logger.INFO, f"Building TRT engine to {engine_file}")

    config_args = {
        "precision": dtype,
        "strongly_typed": False,
        "model_type": 'qwen_2_5_vl',
    }

    config_wrapper = Builder().create_builder_config(**config_args)
    config = config_wrapper.trt_builder_config

    parser = trt.OnnxParser(network, logger)

    with open(onnx_file, 'rb') as model:
        if not parser.parse(model.read(), os.path.abspath(onnx_file)):
            logger.log(trt.Logger.ERROR, "Failed parsing %s" % onnx_file)
            for error in range(parser.num_errors):
                logger.log(trt.Logger.ERROR, parser.get_error(error))
            exit()
        logger.log(trt.Logger.INFO, "Succeeded parsing %s" % onnx_file)
    
    config.add_optimization_profile(profile)
    
    engine_string = builder.build_serialized_network(network, config)
    if engine_string is None:
        raise RuntimeError("Failed building %s" % (engine_file))
    else:
        logger.log(trt.Logger.INFO,
                    "Succeeded building %s " % engine_file)
        with open(engine_file, 'wb') as f:
            f.write(engine_string)
    Builder.save_config(config_wrapper, config_file)
                       

model = ModelWrapper().to('cuda')
model.forward = model.forward_navie

torch.save(model.state_dict(), 'custom_model.pth')
model.load_state_dict(torch.load('custom_model.pth'))
# model = model.half()
torch_dtype = torch.float16
hidden_states = torch.load('hidden_states.pt')
rotary_pos_emb = torch.load('rotary_pos_emb.pt').to(torch_dtype)
full_attention_mask = torch.load('full_attention_mask.pt').to(torch_dtype)
window_attention_mask = torch.load('window_attention_mask.pt').to(torch_dtype)

# full_attention_mask = window_attention_mask

model(hidden_states, full_attention_mask, rotary_pos_emb)

# export ONNX
op = torch.onnx.export(
    model,
    (hidden_states, full_attention_mask, rotary_pos_emb),
    "custom_model.onnx",
    input_names=['input', 'full_attention_mask', 'rotary_pos_emb'],
    output_names=['output'],
    opset_version=17,
)

model.eval()

## export trt
generate_trt('float16')

# print('input is \n', dummy_input)
## run torch
print('torch output is \n', torch_forward(model, hidden_states, full_attention_mask, rotary_pos_emb))

## run onnx
# print('onnx forward is \n', onnx_forward(hidden_states, full_attention_mask, rotary_pos_emb))

### run trt
# print('trt qwen output is \n', trt_forward_qwen('float32'))
print('trt qwen output is \n', trt_forward([hidden_states, full_attention_mask, rotary_pos_emb], 'float16'))
