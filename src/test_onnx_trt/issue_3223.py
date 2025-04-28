import os

import torch
import torch.nn as nn

from typing import Any, Dict, List, Optional, Tuple, Union

import tensorrt as trt
from tensorrt_llm._utils import torch_dtype_to_str, to_json_file
from tensorrt_llm.builder import Builder
from tensorrt_llm.logger import logger
from tensorrt_llm.runtime.session import Session, TensorInfo

from tensorrt_llm._utils import (mpi_rank, str_dtype_to_torch, str_dtype_to_trt,
                        supports_inflight_batching, torch_dtype_to_trt,
                        trt_dtype_to_torch)


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
    
    
class MyFA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, len_q, len_k, max_q, max_k) -> torch.Tensor:
        # ctx.save_for_backward(input)
        return flash_attn_varlen_func(q, k, v, len_q, len_k, max_q, max_k)

    @staticmethod
    def symbolic(g: torch.Graph, q, k, v, len_q, len_k, max_q, max_k) -> torch.Value:
        return g.op("MyFA", q, k, v, len_q, len_k, max_q, max_k)

class MyModel(nn.Module):
    # def __init__(self, bias: bool = False):
    #     super().__init__()
    #     self.hidden_size = 768
    #     self.intermediate_size = 1024
    #     self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
    #     self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=bias)
    #     self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=bias)
    #     self.act_fn = nn.SiLU()

    # def forward(self, hidden_state):
    #     return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))
    
    def __init__(self, dim: int = 1280, num_heads: int = 16) -> None:
            super().__init__()
            self.num_heads = num_heads
            self.head_dim = dim // num_heads
            self.qkv = nn.Linear(dim, dim * 3, bias=True)
            self.proj = nn.Linear(dim, dim)
            
            self.cu_seqlens = torch.load('/sys/fs/cgroup/zzz_tmp/TensorRT-LLM/examples/multimodal/flash_attn_trt/cu_seqlens.pt').to('cuda')
            self.cu_window_seqlens = torch.load('/sys/fs/cgroup/zzz_tmp/TensorRT-LLM/examples/multimodal/flash_attn_trt/cu_window_seqlens.pt').to('cuda')
            
    ## eager attn
    def forward_0(self,
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
    ## flash attn
    def forward(
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
            
        # q, k = apply_rotary_pos_emb_flashatt(q.unsqueeze(0), k.unsqueeze(0), cos, sin)
        q = apply_rotary_pos_emb_vision(q.unsqueeze(0),
                                        rotary_pos_emb).squeeze(0)
        k = apply_rotary_pos_emb_vision(k.unsqueeze(0),
                                        rotary_pos_emb).squeeze(0)
        
        q = q.squeeze(0)
        k = k.squeeze(0)

        # if attention_mask[0][0][-1] < 0:
        #     # window atten
        #     cu_seqlens = self.cu_window_seqlens
        # else:
        #     cu_seqlens = self.cu_seqlens
        cu_seqlens = self.cu_window_seqlens
        
        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        
        dtype = q.dtype
        q = q.to(torch.bfloat16)
        k = k.to(torch.bfloat16)
        v = v.to(torch.bfloat16)
        # attn_output = flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen).reshape(
        #     seq_length, -1
        # )
        attn_output = MyFA.apply(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen).reshape(seq_length, -1)
        attn_output = attn_output.to(dtype)
        
        attn_output = self.proj(attn_output)
        # attn_output = self.proj(hidden_states)
        return attn_output
    

def torch_forward(model: nn.Module, hidden_states, full_attention_mask, rotary_pos_emb):
    return model(hidden_states, full_attention_mask, rotary_pos_emb)

def onnx_forward(hidden_states, full_attention_mask, rotary_pos_emb):
    import onnxruntime as ort
    import numpy as np

    model_path = "custom_model.onnx"
    session = ort.InferenceSession(model_path)

    input_names = [input.name for input in session.get_inputs()]
    output_names = [output.name for output in session.get_outputs()]

    
    hidden_states = hidden_states.detach().cpu().numpy()
    full_attention_mask = full_attention_mask.detach().cpu().numpy()
    rotary_pos_emb = rotary_pos_emb.detach().cpu().numpy()

    # inputs = {input_names[0]: hidden_states, input_names[1]: full_attention_mask, input_names[2]: rotary_pos_emb}
    # inputs = {input_names[0]: hidden_states}
    inputs = {}

    outputs = session.run(output_names, inputs)

    output_data = outputs[0]
    return output_data

def trt_forward(x: torch.Tensor, dtype: str='float32'):
    # engine_file = 'model_FP16.plan'
    engine_file = 'custom_model.engine'
    with open(engine_file, 'rb') as f:
        engine_buffer = f.read()

    visual_encoder_session = Session.from_serialized_engine(
        engine_buffer)
    
    dummy_input = x
    visual_features = {
        'input': dummy_input.to(str_dtype_to_torch(dtype)),
    }
    dummy_input = dummy_input.to(str_dtype_to_torch(dtype))
    tensor_info = [
        TensorInfo('input', str_dtype_to_trt(dtype), dummy_input.shape),
    ]
    visual_output_info = visual_encoder_session.infer_shapes(
        tensor_info)
    visual_encoder_session.set_shapes(visual_features)
    visual_outputs = {
        t.name:
        torch.empty(tuple(t.shape),
                    dtype=trt_dtype_to_torch(t.dtype),
                    device=dummy_input.device)
        for t in visual_output_info
    }
    stream = torch.cuda.Stream(torch.cuda.current_device())
    torch.cuda.set_stream(stream)
    ok = visual_encoder_session.run(visual_features, visual_outputs, stream.cuda_stream)
    assert ok, "Runtime execution failed for vision encoder session"
    image_embeds = visual_outputs['output']
    return image_embeds

def trt_forward_qwen(dtype: str='float32'):
    engine_file = 'model_FP16.plan'
    with open(engine_file, 'rb') as f:
        engine_buffer = f.read()

    visual_encoder_session = Session.from_serialized_engine(
        engine_buffer)
    
    hidden_states = torch.load('/sys/fs/cgroup/zzz_tmp/TensorRT-LLM/examples/multimodal/flash_attn_trt/hidden_states.pt')
    rotary_pos_emb = torch.load('/sys/fs/cgroup/zzz_tmp/TensorRT-LLM/examples/multimodal/flash_attn_trt/rotary_pos_emb.pt')
    full_attention_mask = torch.load('/sys/fs/cgroup/zzz_tmp/TensorRT-LLM/examples/multimodal/flash_attn_trt/full_attention_mask.pt')
    window_attention_mask = torch.load('/sys/fs/cgroup/zzz_tmp/TensorRT-LLM/examples/multimodal/flash_attn_trt/window_attention_mask.pt')
    
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

    # onnx_file = f'/sys/fs/cgroup/zzz_tmp/TensorRT-LLM/examples/multimodal/flash_attn_trt/tmp/trt_engines/Qwen2_5-VL/fp16/1-gpu/vision/onnx/model.onnx'
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
                       

model = MyModel().to('cuda')
# torch.save(model.state_dict(), 'custom_model.pth')
model.load_state_dict(torch.load('custom_model.pth'))
# model = model.half()
hidden_states = torch.load('/sys/fs/cgroup/zzz_tmp/TensorRT-LLM/examples/multimodal/flash_attn_trt/hidden_states.pt')
rotary_pos_emb = torch.load('/sys/fs/cgroup/zzz_tmp/TensorRT-LLM/examples/multimodal/flash_attn_trt/rotary_pos_emb.pt')
full_attention_mask = torch.load('/sys/fs/cgroup/zzz_tmp/TensorRT-LLM/examples/multimodal/flash_attn_trt/full_attention_mask.pt')
window_attention_mask = torch.load('/sys/fs/cgroup/zzz_tmp/TensorRT-LLM/examples/multimodal/flash_attn_trt/window_attention_mask.pt')


model(hidden_states, full_attention_mask, rotary_pos_emb)

# export ONNX
torch.onnx.export(
    model,
    (hidden_states, full_attention_mask, rotary_pos_emb),
    "custom_model.onnx",
    input_names=['input', 'full_attention_mask', 'rotary_pos_emb'],
    output_names=['output'],
    opset_version=19,
)

## export trt
# generate_trt('float32')

# print('input is \n', dummy_input)
## run torch
print('torch output is \n', torch_forward(model, hidden_states, full_attention_mask, rotary_pos_emb))

## run onnx
print('onnx forward is \n', onnx_forward(hidden_states, full_attention_mask, rotary_pos_emb))

### run trt
# print('trt qwen output is \n', trt_forward_qwen('float16'))
# print('trt qwen output is \n', trt_forward(dummy_input, 'float32'))
