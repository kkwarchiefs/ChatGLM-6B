LR=1e-4
PRE_SEQ_LEN=128
DATESTR=$(date +"%m-%d-%H-%M")
MASTER_PORT=$(shuf -n 1 -i 10000-65535)


nohup python3 -m torch.distributed.launch --nproc_per_node 8 main.py \
    --do_train \
    --train_file /search/ai/jamsluo/chatglm_6b/dataset/mrc_all_0705_4k/train.json \
    --test_file /search/ai/jamsluo/chatglm_6b/dataset/mrc_all_0705_4k/dev.json \
    --prompt_column content \
    --response_column summary \
    --model_name_or_path /search/ai/pretrain_models/chatglm-6b-1.1/snapshots/a10da4c68b5d616030d3531fc37a13bb44ea814d/ \
    --output_dir ./output/mrc-409-explain-$PRE_SEQ_LEN-$LR-$DATESTR \
    --overwrite_output_dir \
    --max_source_length 4096 \
    --max_target_length 512 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --predict_with_generate \
    --max_steps 2000 \
    --logging_steps 10 \
    --save_steps 500 \
    --learning_rate $LR \
    --pre_seq_len $PRE_SEQ_LEN \
    --fp16  > logs/log.log-$DATESTR &
