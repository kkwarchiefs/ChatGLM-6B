LR=5e-5
DATESTR=$(date +"%m-%d-%H-%M")
MASTER_PORT=$(shuf -n 1 -i 10000-65535)


nohup  deepspeed --num_gpus=8 --master_port $MASTER_PORT main.py \
    --deepspeed deepspeed.json \
    --do_train \
    --train_file /search/ai/jamsluo/chatglm_6b/dataset/mrc_0712_4k/train.json \
    --test_file /search/ai/jamsluo/chatglm_6b/dataset/mrc_all_0705_4k/dev.json \
    --prompt_column content \
    --response_column summary \
    --model_name_or_path /search/ai/pretrain_models/chatglm-6b-1.1/snapshots/a10da4c68b5d616030d3531fc37a13bb44ea814d/ \
    --output_dir ./output/mrc-4096-explain-fineturning-0712-$LR-$DATESTR \
    --overwrite_output_dir \
    --max_source_length 4096 \
    --max_target_length 512 \
    --per_device_train_batch_size 3 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --predict_with_generate \
    --max_steps 2000 \
    --logging_steps 10 \
    --save_steps 500 \
    --learning_rate $LR \
    --lr_scheduler_type "cosine" \
    --fp16  > logs/log.log-$DATESTR &
