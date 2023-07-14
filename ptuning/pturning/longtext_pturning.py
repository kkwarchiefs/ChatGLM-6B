#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : longtext.py
# @Author: 罗锦文
# @Date  : 2023/7/4
# @Desc  : 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import codecs
import torch
import os

if __name__ == "__main__":
    from transformers import AutoTokenizer, AutoModel, AutoConfig
    tokenizer = AutoTokenizer.from_pretrained("/search/ai/pretrain_models/chatglm-6b-1.1/snapshots/a10da4c68b5d616030d3531fc37a13bb44ea814d", trust_remote_code=True)
    config = AutoConfig.from_pretrained("/search/ai/jamsluo/chatglm_6b/output/mrc-8192-explain-128-1e-4-07-07-12-21/checkpoint-2000/", trust_remote_code=True)
    model = AutoModel.from_pretrained("/search/ai/pretrain_models/chatglm-6b-1.1/snapshots/a10da4c68b5d616030d3531fc37a13bb44ea814d", trust_remote_code=True, config=config)
    prefix_state_dict = torch.load(os.path.join("/search/ai/jamsluo/chatglm_6b/output/mrc-8192-explain-128-1e-4-07-07-12-21/checkpoint-2000/", "pytorch_model.bin"))
    new_prefix_state_dict = {}
    for k, v in prefix_state_dict.items():
        if k.startswith("transformer.prefix_encoder."):
            new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
    model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)
    model = model.half().cuda().eval()
    text = open(sys.argv[1], "r").read()
    print(len(tokenizer(text)['input_ids']))
    while True:
        query = input("\n请输入问题, 输入exit结束：")
        if query == "exit":
            break
        response, history = model.chat(tokenizer, text + "<n>根据上面内容回答问题：<n>" + query, history=[], max_length=5120)
        print(response)
