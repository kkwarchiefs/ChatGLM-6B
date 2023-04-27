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
import transformers
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    set_seed,
)
from peft import get_peft_model, LoraConfig, TaskType
def convert_model():
    RM_model_path = sys.argv[1]
    model_dict = torch.load(os.path.join(RM_model_path, 'pytorch_model-00001-of-00008.bin'), map_location="cpu")
    # model_dict = torch.load(os.path.join(RM_model_path, 'pytorch_model.bin'), map_location="cpu")

    for k, v in model_dict.items():
        print(k)
        if 'word_embeddings.weight' in k:
            print(v[130001, :5])

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

def merge_lora():
    config = AutoConfig.from_pretrained("/search/ai/pretrain_models/chatglm-6b/", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("/search/ai/pretrain_models/chatglm-6b/", trust_remote_code=True)
    model = AutoModel.from_pretrained(sys.argv[1], trust_remote_code=True)
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=64,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
    )

    model = get_peft_model(model, peft_config)
    model_dict = torch.load(os.path.join(sys.argv[1], 'pytorch_model.bin'))
    new_model_dict = {}
    for key, value in model_dict.items():
        if 'lora_' in key:
            key = key.replace('.weight', '.default.weight')
        new_model_dict[key] = value
    model.load_state_dict(new_model_dict, strict=True)
    model = model.merge_and_unload()

    lora_model_sd = model.state_dict()
    deloreanized_sd = {
        k.replace("base_model.model.", ""): v
        for k, v in lora_model_sd.items()
        if "lora" not in k
    }

    # ChatGLMForConditionalGeneration.save_pretrained(
    #     base_model, output_dir, state_dict=deloreanized_sd
    # )

    model.save_pretrained(
        sys.argv[2], state_dict=deloreanized_sd, max_shard_size="1900MB"
    )
    tokenizer.save_pretrained(sys.argv[2])

if __name__ == "__main__":
    convert_model()
