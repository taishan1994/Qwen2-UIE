MODEL_PATH="model_hub/qwen2_0.5B_instruct"
NPROC_PER_NODE=8
NNODES=1
NODE_RANK=0
MASTER_ADDR="0.0.0.0"
MASTER_PORT=8896

DS_CONFIG_PATH="examples/deepspeed/ds_z2_config.json"
OUTPUT_PATH="output"

DISTRIBUTED_ARGS="
    --nproc_per_node $NPROC_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
  "

NCCL_IB_DISABLE=1, NCCL_P2P_DISABLE=1 torchrun $DISTRIBUTED_ARGS src/train.py \
    --deepspeed $DS_CONFIG_PATH \
    --stage sft \
    --do_train \
    --use_fast_tokenizer \
    --flash_attn "auto" \
    --model_name_or_path $MODEL_PATH \
    --dataset alpaca_zh_demo \
    --template qwen \
    --finetuning_type lora \
    --lora_target q_proj,v_proj\
    --output_dir $OUTPUT_PATH \
    --overwrite_cache \
    --overwrite_output_dir \
    --warmup_steps 100 \
    --weight_decay 0.1 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --ddp_timeout 9000 \
    --learning_rate 5e-6 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --cutoff_len 4096 \
    --save_steps 1000 \
    --plot_loss \
    --num_train_epochs 100 \
    --bf16
