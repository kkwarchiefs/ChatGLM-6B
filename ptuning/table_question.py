import os
import sys
import json
import openai
import random

openai.api_key = "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"

sum_piece_old = "Give me 3 Chinese questions based on the cell of the table . The response should be in the standard JSON format, as follows: ```json[\"<question>\",\"<question>\",\"<question>\"]``` Replace <question> with your answer. The answer should be concise enough, without unnecessary explanations, and avoid adding any numbering like \"问题 1\", etc."
table_fout = open(sys.argv[1], 'w')

# for index, line in enumerate(sys.stdin):
#     ins = json.loads(line)
#     input_word = ins['parts']
#     messages = [
#         {
#             "role": "user",
#             "content": "I have an MarkDown table whose content is that \"" + input_word + sum_piece
#         }]
#     try:
#         response = openai.ChatCompletion.create(
#             model="gpt-3.5-turbo",
#             messages=messages,
#             temperature=0.2
#         )["choices"][0]["message"]["content"]
#     except:
#         print ("error")
#         response = "error"
#     ins['details'] = response
#     table_fout.write(json.dumps(ins, ensure_ascii=False)+'\n')
#     table_fout.flush()
# table_fout.close()

for index, line in enumerate(sys.stdin):
    ins = json.loads(line)
    input_word = ins['content']
    messages = [
        {
            "role": "user",
            "content": input_word.strip()
        }]
    token_length = ins['length']
    if token_length < 3600:
        modelstr = "gpt-3.5-turbo"
    else:
        modelstr = "gpt-3.5-turbo-16k"
    try:
        response = openai.ChatCompletion.create(
            # model="gpt-3.5-turbo",
            model=modelstr,
            messages=messages,
            temperature=0.0
        )["choices"][0]["message"]["content"]
    except Exception as e:
        print ("error", e)
        response = "error"
    ins['response'] = response
    table_fout.write(json.dumps(ins, ensure_ascii=False)+'\n')
    table_fout.flush()
table_fout.close()


