LR=1e-4
PRE_SEQ_LEN=128
DATESTR=$(date +"%m-%d-%H-%M")
MASTER_PORT=$(shuf -n 1 -i 10000-65535)

 nohup deepspeed --num_gpus=8 --master_port $MASTER_PORT main.py \
    --deepspeed deepspeed.json \
    --do_train \
    --train_file /search/ai/jamsluo/chatglm_6b/dataset/mrc_all_0705_4k/train.json \
    --validation_file /search/ai/jamsluo/chatglm_6b/dataset/mrc_all_0705_4k/dev.json \
    --preprocessing_num_workers 10 \
    --prompt_column content \
    --response_column summary \
    --overwrite_cache \
    --model_name_or_path /search/ai/pvopliu/chatglm-6b/chatglm2/snapshots/a6d54fac46dff2db65d53416c207a4485ca6bd40/ \
    --output_dir ./output/mrc-4096-fine-full-$LR-$DATESTR \
    --overwrite_output_dir \
    --max_source_length 4096 \
    --max_target_length 512 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --predict_with_generate \
    --max_steps 2000 \
    --logging_steps 10 \
    --save_steps 100 \
    --lr_scheduler_type "cosine" \
    --learning_rate $LR  > logs/log.log-$DATESTR &
