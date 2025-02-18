MODEL="RWKV"
LOAD_MODEL={rwkv_hf_dir}
MODEL_ARGS="model=$MODEL,load_model=${LOAD_MODEL},temperature=0,max_model_length=32768,max_new_tokens=1024,batch_size=64"
# TASK=aime24
TASK=math_500
# TASK="gpqa:diamond"
OUTPUT_DIR=./evals/$MODEL/$TASK

lighteval rwkv $MODEL_ARGS "custom|$TASK|0|0" \
    --custom-tasks lighteval.py \
    --system-prompt="Please reason step by step, and put your final answer within \boxed{}." \
    --output-dir $OUTPUT_DIR
