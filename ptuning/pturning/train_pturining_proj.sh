LR=1e-3
PRE_SEQ_LEN=128
DATESTR=$(date +"%m-%d-%H-%M")
MASTER_PORT=$(shuf -n 1 -i 10000-65535)


nohup  python3 -m torch.distributed.launch --nproc_per_node 8 main.py \
    --do_train \
    --train_file dataset/rc_0422_new/train.json \
    --test_file dataset/rc_0422_new/dev.json \
    --prompt_column content \
    --response_column summary \
    --model_name_or_path /search/ai/pretrain_models/chatglm-6b/ \
    --output_dir ./output/rc-1536-chatglm-6b-proj-$PRE_SEQ_LEN-$LR-$DATESTR \
    --overwrite_output_dir \
    --max_source_length 1536 \
    --max_target_length 512 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --predict_with_generate \
    --max_steps 6000 \
    --logging_steps 10 \
    --save_steps 500 \
    --learning_rate $LR \
    --pre_seq_len $PRE_SEQ_LEN \
    --prefix_projection \
    --fp16 > logs/log.log-$DATESTR &
