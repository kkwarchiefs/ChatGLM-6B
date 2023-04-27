#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : convert_model.py
# @Author: 罗锦文
# @Date  : 2023/3/14
# @Desc  : 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import codecs
import torch
import os

def convert_model():
    RM_model_path = "./output/rm_model/"
    model_dict = torch.load(os.path.join(RM_model_path, 'pytorch_model.bin'), map_location="cpu")
    new_model_dict = {k.replace('hf_model.', ''): v for k, v in model_dict.items()}
    torch.save(new_model_dict, os.path.join(RM_model_path, 'model.pt'))

def covert_onnx():
    from onnx import load_model, save_model
    import torch
    import torch.nn as nn
    from onnxmltools.utils import float16_converter
    import numpy as np
    path = sys.argv[1]
    out = sys.argv[2]
    onnx_model = load_model(path + '/model.onnx')
    new_onnx_model = float16_converter.convert_float_to_float16(onnx_model, keep_io_types=True)
    save_model(new_onnx_model, out + '/model_fp16.onnx')

if __name__ == "__main__":
    covert_onnx()
