#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : merge_context.py
# @Author: 罗锦文
# @Date  : 2023/7/5
# @Desc  : 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import codecs
import json
from collections import defaultdict
def read_res():
    query2count = defaultdict(int)
    query2res = {}
    for line, res in zip(open('./datas/web_source_0625_0.csv'), open("./datas/web_target_0625_0.csv")):
        text = line.strip()
        if '<n>根据上面内容回答问题：<n>' not in text:
            continue
        if res == 'error':
            continue
        items = text.split('<n>根据上面内容回答问题：<n>')
        query2count[items[1]] += 1
        query2res[items[1]] = res
    for line, res in zip(open('./datas/web_source_0625_1.csv'), open("./datas/web_target_0625_1.csv")):
        text = line.strip()
        if '<n>根据上面内容回答问题：<n>' not in text:
            continue
        if res == 'error':
            continue
        items = text.split('<n>根据上面内容回答问题：<n>')
        query2count[items[1]] += 1
        query2res[items[1]] = res
    query2unique = {k: v for k, v in query2count.items() if v == 1}
    # for k, v in query2count.items():
    #     if v > 1:
    #         query2count.pop(k)
    return query2unique, query2res

def merge():
    query2count, query2res = read_res()
    print(len(query2count), len(query2res), file=sys.stderr, flush=True)
    datas = open("./datas/web_page_0620.csv").read().splitlines()
    q_data = open("./datas/web_page_question_target_0620.csv").read().splitlines()
    print(len(datas), len(q_data), file=sys.stderr, flush=True)
    for line, queries in zip(datas, q_data):
        text = line.strip()
        try:
            queries = json.loads(queries)
        except:
            continue
        for q in queries:
            if q in query2count:
                ins = {
                    "content": text + '<n>根据上面内容回答问题：<n>' + q,
                    "summary": query2res[q].strip()
                }
                print(json.dumps(ins, ensure_ascii=False))

def merge():
    query2count, query2res = read_res()
    print(len(query2count), len(query2res), file=sys.stderr, flush=True)
    datas = open("./datas/web_page_0620.csv").read().splitlines()
    q_data = open("./datas/web_page_question_target_0620.csv").read().splitlines()
    print(len(datas), len(q_data), file=sys.stderr, flush=True)
    for line, queries in zip(datas, q_data):
        text = line.strip()
        try:
            queries = json.loads(queries)
        except:
            continue
        for q in queries:
            if q in query2count:
                ins = {
                    "content": text + '<n>根据上面内容回答问题：<n>' + q,
                    "summary": query2res[q].strip()
                }
                print(json.dumps(ins, ensure_ascii=False))

def find_long():
    tokenizer = AutoTokenizer.from_pretrained("/search/ai/pretrain_models/chatglm-6b-1.1/snapshots/a10da4c68b5d616030d3531fc37a13bb44ea814d", trust_remote_code=True)
    query2count, query2res = read_res()
    print(len(query2count), len(query2res), file=sys.stderr, flush=True)
    datas = open("./datas/web_page_0620.csv").read().splitlines()
    q_data = open("./datas/web_page_question_target_0620.csv").read().splitlines()
    print(len(datas), len(q_data), file=sys.stderr, flush=True)
    for line, queries in zip(datas, q_data):
        text = line.strip()
        try:
            queries = json.loads(queries)
        except:
            continue

        for q in queries:
            if q in query2count:
                ins = {
                    "content": text + '<n>根据上面内容回答问题：<n>' + q,
                    "summary": query2res[q].strip()
                }
                print(json.dumps(ins, ensure_ascii=False))
def cal_len():
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained("/search/ai/pretrain_models/chatglm-6b-1.1/snapshots/a10da4c68b5d616030d3531fc37a13bb44ea814d", trust_remote_code=True)
    for line in sys.stdin:
        ins = json.loads(line)
        if len(tokenizer(ins['content'])['input_ids']) < 4096:
            if type(ins['content']) == str and type(ins['content']) == str:
                print(line.strip())

if __name__ == "__main__":
    # merge()
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained("/cfs/cfs-g22qkwzd/jamsluo/chatpdf/models/chatglm-6b-1.1/", trust_remote_code=True)
    for line in sys.stdin:
        ins = json.loads(line)
        if ins['length_chat'] > 4096:
            continue
        if len(tokenizer(ins['response'])['input_ids']) > 512:
            continue
        res = {
            'content': ins["content"].replace('\n', '<n>'),
            'summary': ins['response'].replace('\n', '<n>')
        }
        print(json.dumps(res, ensure_ascii=False))
