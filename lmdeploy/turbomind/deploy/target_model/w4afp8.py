# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import sys

import torch

import lmdeploy

from ..source_model.base import BaseInputModel, BaseReader
from .base import (OUTPUT_MODELS, BaseOutputModel, TurbomindModelConfig,
                   merge_qkv, permute)

# import _turbomind as _tm
# TODO: find another way import _turbomind
lmdeploy_dir = osp.split(lmdeploy.__file__)[0]
sys.path.append(osp.join(lmdeploy_dir, 'lib'))
import _turbomind as _tm  # noqa: E402

def get_cuda_tensor(tensors):
    """Get cuda tensor."""
    result = map(lambda x: x.cuda() if x is not None else x, tensors)
    return (*result, )

@OUTPUT_MODELS.register_module(name='w4afp8')
class TurbomindW4Afp8Model(BaseOutputModel):
    """Export to turbomind w4a16 format."""

    def __init__(self,
                 input_model: BaseInputModel,
                 cfg: TurbomindModelConfig,
                 to_file: bool = True,
                 out_dir: str = ''):
        super().__init__(input_model, cfg, to_file, out_dir)

    def export_transformer_block(self, bin: BaseReader, i: int):
        """Export transformer layer i."""
        group_size = self.cfg.group_size
        tp = self.cfg.tensor_para_size
        size_per_head = self.cfg.size_per_head
        # attn
        qkv_qw, o_qw = get_cuda_tensor(bin.attn(i))
        qkv_s, o_s = get_cuda_tensor(bin.attn_scale(i))
        qkv_is, o_is = get_cuda_tensor(bin.attn_input_scale(i))
        qkv_a, o_a = get_cuda_tensor(bin.attn_alhpa(i))

        # TODO all split_dim should be ensure for case tp >1
        self.save_split(qkv_qw, f'layers.{i}.attention.w_qkv.qweight', -1)
        self.save_split(qkv_s, f'layers.{i}.attention.w_qkv.scales_zeros', -1)
        self.save_split(qkv_is, f'layers.{i}.attention.w_qkv.input_scales', -1)
        self.save_split(qkv_a, f'layers.{i}.attention.w_qkv.alpha', copy=True)

        self.save_split(o_qw, f'layers.{i}.attention.wo.qweight', 0)
        self.save_split(o_s, f'layers.{i}.attention.wo.scales_zeros', 0)
        self.save_split(o_is, f'layers.{i}.attention.wo.input_scales', -1)
        self.save_split(o_a, f'layers.{i}.attention.wo.alpha', copy=True)

        # ffn weights
        # TODO fuse act(w1(x)) * w(3) kernel not imped, do not fuse right now
        w1_qw, w2_qw, w3_qw = get_cuda_tensor(bin.ffn(i))
        w1_s, w2_s, w3_s = get_cuda_tensor(bin.ffn_scale(i))
        w1_is, w2_is, w3_is = get_cuda_tensor(bin.attn_input_scale(i))
        w1_a, w2_a, w3_a = get_cuda_tensor(bin.attn_alhpa(i))
        
        self.save_split(w1_qw, f'layers.{i}.feed_forward.w1.qweight', -1)
        self.save_split(w1_s, f'layers.{i}.feed_forward.w1.scales_zeros', -1)
        self.save_split(w1_is, f'layers.{i}.feed_forward.w1.input_scales', -1)
        self.save_split(w1_a, f'layers.{i}.feed_forward.w1.alpha', copy=True)

        self.save_split(w3_qw, f'layers.{i}.feed_forward.w3.qweight', -1)
        self.save_split(w3_s, f'layers.{i}.feed_forward.w3.scales_zeros', -1)
        self.save_split(w3_is, f'layers.{i}.feed_forward.w3.input_scales', -1)
        self.save_split(w3_a, f'layers.{i}.feed_forward.w3.alpha', copy=True)

        self.save_split(w2_qw, f'layers.{i}.feed_forward.w2.qweight', -1)
        self.save_split(w2_s, f'layers.{i}.feed_forward.w2.scales_zeros', -1)
        self.save_split(w2_is, f'layers.{i}.feed_forward.w2.input_scales', -1)
        self.save_split(w2_a, f'layers.{i}.feed_forward.w2.alpha', copy=True)

        # norm
        attn_norm = bin.attn_norm(i)
        ffn_norm = bin.ffn_norm(i)
        self.save_split(attn_norm, f'layers.{i}.attention_norm.weight')
        self.save_split(ffn_norm, f'layers.{i}.ffn_norm.weight')
