LR=1e-5
PRE_SEQ_LEN=128
DATESTR=$(date +"%m-%d-%H-%M")
MASTER_PORT=$(shuf -n 1 -i 10000-65535)

nohup deepspeed --num_gpus=8 --master_port $MASTER_PORT main.py \
    --deepspeed deepspeed.json \
    --do_train \
    --train_file dataset/rc_0422_new/train.json \
    --test_file dataset/rc_0422_new/dev.json \
    --prompt_column content \
    --response_column summary \
    --model_name_or_path /search/ai/pretrain_models/chatglm-6b/ \
    --output_dir ./output/rc-1536-chatglm-lowlr-ptuning-$PRE_SEQ_LEN-$LR-$DATESTR \
    --overwrite_output_dir \
    --max_source_length 1536 \
    --max_target_length 512 \
    --per_device_train_batch_size 12 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --predict_with_generate \
    --max_steps 3000 \
    --logging_steps 10 \
    --save_steps 500 \
    --learning_rate $LR \
    --pre_seq_len $PRE_SEQ_LEN \
    --lr_scheduler_type "cosine" \
    --fp16 > logs/log.log-$DATESTR &
