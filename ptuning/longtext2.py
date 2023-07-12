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

if __name__ == "__main__":
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained("/search/ai/pvopliu/chatglm-6b/chatglm2/snapshots/a6d54fac46dff2db65d53416c207a4485ca6bd40/", trust_remote_code=True)
    # tokenizer = AutoTokenizer.from_pretrained("/search/ai/pvopliu/chatglm-6b/chatglm-6b-1.1/snapshots/a10da4c68b5d616030d3531fc37a13bb44ea814d/", trust_remote_code=True)

    # print(tokenizer.bos_token_id)
    # print(tokenizer(['你好', '你好你好你好你好你好'], max_length=30, truncation=True, return_tensors="pt", padding=True))
    # model = AutoModel.from_pretrained("/search/ai/pvopliu/chatglm-6b/chatglm2/snapshots/a6d54fac46dff2db65d53416c207a4485ca6bd40/", trust_remote_code=True)
    model = AutoModel.from_pretrained(
        "/search/ai/jamsluo/ChatGLM2-6B/ptuning/output/mrc-4096-fine-full-1e-4-07-10-20-14/checkpoint-1100/",
        trust_remote_code=True).half().cuda()
    model = model.eval()
    text = open(sys.argv[1], "r").read()
    while True:
        query = input("\n请输入问题, 输入exit结束：")
        if query == "exit":
            break
        response, history = model.chat(tokenizer, text + "<n>根据上面内容回答问题：<n>" + query, history=[], max_length=5120)
        print(response)
        # 快速排序


