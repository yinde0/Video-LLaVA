JSON_FOLDER="$HOME/pt_json/train_json"
IMAGE_FOLDER="/ephemeral/data/llava_image_tune_2"
VIDEO_FOLDER="/ephemeral/data/videochatgpt_tune_2"

cd "$HOME/code/Video-LLaVA"

OUT_DIR="/ephemeral/checkpoints/videollava-7b"



export PYTHONPATH="$PWD:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:256"
export NCCL_SOCKET_IFNAME=$(ip -o -4 route show to default | awk '{print $5}')
export NCCL_IB_DISABLE=1
export NCCL_NET_GDR_LEVEL=0
export HF_HOME=/ephemeral/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export TORCH_HOME=/ephemeral/.cache/torch

deepspeed videollava/train/train_mem.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed scripts/zero2_offload.json \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --version v1 \
    --data_path ${JSON_FOLDER}/llava_image_tune_.json ${JSON_FOLDER}/videochatgpt_tune_.json ${JSON_FOLDER}/nlp_tune.json \
    --image_folder ${IMAGE_FOLDER} \
    --image_tower LanguageBind/LanguageBind_Image \
    --video_folder ${VIDEO_FOLDER} \
    --video_tower LanguageBind/LanguageBind_Video_merge \
    --mm_projector_type mlp2x_gelu \
    --pretrain_mm_mlp_adapter ./checkpoints/videollava-7b-pretrain/mm_projector.bin \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ${OUT_DIR} \
    --resume_from_checkpoint ./checkpoints/videollava-7b-lora \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --max_steps 600 \
    --save_steps 600 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 1024  --tokenizer_model_max_length 1024 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to tensorboard  \
    --cache_dir "./cache_dir"
