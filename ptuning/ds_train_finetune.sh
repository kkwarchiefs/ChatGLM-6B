
LR=1e-4
DATESTR=$(date +"%m-%d-%H-%M")
MASTER_PORT=$(shuf -n 1 -i 10000-65535)

nohup deepspeed --num_gpus=8 --master_port $MASTER_PORT main.py \
    --deepspeed deepspeed.json \
    --do_train \
    --train_file dataset/rc_0422_new/train.json \
    --test_file dataset/rc_0422_new/dev.json \
    --prompt_column content \
    --response_column summary \
    --model_name_or_path /search/ai/pretrain_models/chatglm-6b-dropout/ \
    --output_dir ./output/rc-1536-chatglm-6b-dropout-$DATESTR \
    --overwrite_output_dir \
    --max_source_length 1536 \
    --max_target_length 512 \
    --per_device_train_batch_size 10 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --predict_with_generate \
    --max_steps 6000 \
    --logging_steps 10 \
    --save_steps 500 \
    --learning_rate $LR \
    --lr_scheduler_type "cosine" \
    --fp16 > logs/log.log-$DATESTR &

