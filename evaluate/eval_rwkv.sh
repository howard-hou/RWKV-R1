MODEL={rwkv_hf_dir}
MODEL_ARGS="model=$MODEL,max_model_length=32768,max_new_tokens=2048,batch_size=32"
# TASK=aime24
TASK=math_500
# TASK="gpqa:diamond"
OUTPUT_DIR=./evals/$MODEL/$TASK

lighteval rwkv $MODEL_ARGS "custom|$TASK|0|0" \
    --custom-tasks lighteval.py \
    --use-chat-template \
    --system-prompt="Please reason step by step, and put your final answer within \boxed{}." \
    --output-dir $OUTPUT_DIR
