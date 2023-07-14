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
    tokenizer = AutoTokenizer.from_pretrained("/search/ai/jamsluo/chatglm_6b/output/mrc-8192-explain-lora-1e-4-07-07-18-49/lora_merge/", trust_remote_code=True)
    model = AutoModel.from_pretrained("/search/ai/jamsluo/chatglm_6b/output/mrc-8192-explain-lora-1e-4-07-07-18-49/lora_merge/", trust_remote_code=True).half().cuda()
    model = model.eval()
    text = open(sys.argv[1], "r").read()
    while True:
        query = input("\n请输入问题, 输入exit结束：")
        if query == "exit":
            break
        response, history = model.chat(tokenizer, text + "根据上面文章回答下面问题：" + query, history=[], max_length=5120)
        print(response)
