# Copyright (c) OpenMMLab. All rights reserved.
import torch

from .base import INPUT_MODELS
from .llama import LlamaModel, LlamaReader


def ensure_fp16orint32(tensors: torch.Tensor):
    """Ensure tensors in fp16/int32 format."""
    result = []
    for tensor in tensors:
        if tensor is not None:
            if tensor.dtype in [torch.float16, torch.float32, torch.bfloat16]:
                result.append(tensor.half())
            else:
                assert tensor.dtype == torch.int32
                result.append(tensor)
        else:
            result.append(None)
    return (*result, )


class LlamaW4Afp8Reader(LlamaReader):
    """LlamaW4Afp8Reader."""

    attn_layer_prefix = 'transformer.layers'
    attn_layer_patten = r'transformer.layers.([0-9]+).'
    tok_embeddings_key = 'transformer.vocab_embedding.weight'
    norm_weight_key = 'transformer.ln_f.weight'
    output_weight_key = 'lm_head.weight'

    def __init__(self, new_params: dict, unused_params: dict, last_bin: bool,
                 model_cfg: dict):
        super().__init__(new_params, unused_params, last_bin, model_cfg)



    def _attn(self, i: int, kind: str, allow_none=False):
        """Get q, k, v, o kind for layer i."""
        result = []
        for key in ['qkv', 'dense']:
            tensor = self.params.get(
                f'{self.attn_layer_prefix}.{i}.attention.{key}.{kind}')
            if not allow_none:
                assert tensor is not None
            result.append(tensor)
        return (*result, )

    def attn(self, i: int):
        """Get qkv, o weight for layer i."""
        return self._attn(i, 'weight')

    def attn_scale(self, i: int):
        """Get qkv, o scales for layer i."""
        return ensure_fp16orint32(self._attn(i, 'weights_scaling_factor'))

    def attn_input_scale(self, i: int):
        """Get qkv, o input scales for layer i."""
        return ensure_fp16orint32(self._attn(i, 'prequant_scaling_factor'))

    def attn_alpha(self, i: int):
        """Get qkv, o alpha for layer i."""
        return ensure_fp16orint32(self._attn(i, 'alpha'))

    def attn_norm(self, i: int):
        """Get attn norm for layer i."""
        return self.params[
            f'{self.attn_layer_prefix}.{i}.input_layernorm.weight']

    def _ffn(self, i: int, kind: str):
        """Get ffn kind for layer i."""
        result = []
        for key in ['gate', 'proj', 'fc']:
            tensor = self.params[
                f'{self.attn_layer_prefix}.{i}.mlp.{key}.{kind}']
            result.append(tensor)
        return (*result, )

    def ffn(self, i: int):
        """Get ffn weight for layer i."""
        return self._ffn(i, 'weight')

    def ffn_scale(self, i: int):
        """Get ffn weight scales for layer i."""
        return ensure_fp16orint32(self._ffn(i, 'weights_scaling_factor'))

    def ffn_input_scale(self, i: int):
        """Get ffn input scales for layer i."""
        return ensure_fp16orint32(self._ffn(i, 'prequant_scaling_factor'))

    def ffn_alpha(self, i: int):
        """Get ffn alpha for layer i."""
        return ensure_fp16orint32(self._ffn(i, 'alpha'))

    def ffn_norm(self, i: int):
        """Get ffn norm for layer i."""
        return self.params[
            f'{self.attn_layer_prefix}.{i}.post_layernorm.weight']


@INPUT_MODELS.register_module(name='llama-w4afp8')
class LlamaW4Afp8Model(LlamaModel):
    """Llama Awq model in hf format."""

    Reader = LlamaW4Afp8Reader

    def __init__(self,
                 model_path: str,
                 tokenizer_path: str,
                 ckpt_path: str = None,
                 **kwargs):
        super().__init__(model_path,
                         tokenizer_path,
                         ckpt_path=ckpt_path,
                         **kwargs)
