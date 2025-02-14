export RWKV_V7_ON='1' # ==> enable RWKV-7 mode
export RWKV_JIT_ON='1' # '1' for better speed
export RWKV_CUDA_ON='1' # '1' to compile CUDA kernel (10x faster), requires c++ compiler & cuda libraries
export RWKV_HEAD_SIZE_A="64"
export CUDA_VISIBLE_DEVICES=1

MODEL="RWKV"
LOAD_MODEL={your_rwkv_pt_dir}/xxx.pth
MODEL_ARGS="model=$MODEL,load_model=${LOAD_MODEL},temperature=0,max_model_length=32768,max_new_tokens=1024,num_gpus=1,batch_size=4"
# TASK=aime24
TASK=math_500
# TASK="gpqa:diamond"
OUTPUT_DIR=./evals/$MODEL/$TASK

lighteval rwkv $MODEL_ARGS "custom|$TASK|0|0" \
    --custom-tasks lighteval.py \
    --system-prompt="Please reason step by step, and put your final answer within \boxed{}." \
    --output-dir $OUTPUT_DIR
