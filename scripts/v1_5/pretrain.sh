JSON_FOLDER="$HOME/pt_json/train_json"
IMAGE_FOLDER="/ephemeral/data"
VIDEO_FOLDER="/ephemeral/data"

cd "$HOME/code/Video-LLaVA"
export PYTHONPATH="$PWD:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:256"
export NCCL_SOCKET_IFNAME=$(ip -o -4 route show to default | awk '{print $5}')
export NCCL_IB_DISABLE=1
export NCCL_NET_GDR_LEVEL=0
export HF_HOME=/ephemeral/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export TORCH_HOME=/ephemeral/.cache/torch

deepspeed videollava/train/train_mem.py \
    --deepspeed ./scripts/zero2_accum8.json \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --version v1 \
    --data_path ${JSON_FOLDER}/llava_image_.json ${JSON_FOLDER}/valley_.json \
    --image_folder ${IMAGE_FOLDER} \
    --image_tower LanguageBind/LanguageBind_Image \
    --video_folder ${VIDEO_FOLDER} \
    --video_tower LanguageBind/LanguageBind_Video_merge \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./checkpoints/videollava-7b-pretrain \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 250 \
    --max_steps 250 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 50 \
    --tf32 True \
    --model_max_length 768  --tokenizer_model_max_length 768 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --cache_dir "/ephemeral/.cache"
