

CURR="$HOME/code/Video-LLaVA"
cd  "$CURR"
export PYTHONPATH="$PWD:$PYTHONPATH"


# --- paths ---
CKPT_DIR="checkpoints/video-llava-7b-lora"
QA_ROOT="${CURR}/eval/GPT_Zero_Shot_QA/MSRVTT_Zero_Shot_QA/MSRVTT_Zero_Shot_QA"
CACHE_DIR="${CURR}/cache_dir"
OUT_DIR="${QA_ROOT}/videollava-7b-checkpoint-1"
MODEL_BASE="lmsys/vicuna-7b-v1.5"
mkdir -p "$CACHE_DIR" "$OUT_DIR"

# optional: keep everything local/offline
# export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

# --- GPU sharding over videos ---
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"   # e.g. export CUDA_VISIBLE_DEVICES=0,1,2,3
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
  CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python3  videollava/eval/video/run_inference_video_qa.py  \
    --model_path "${CKPT_DIR}" \
    --model_base "$MODEL_BASE" \
    --video_dir "${QA_ROOT}/videos/all" \
    --gt_file_question "${QA_ROOT}/test_q.json" \
    --gt_file_answers "${QA_ROOT}/test_a.json" \
    --output_dir "${OUT_DIR}" \
    --output_name "${CHUNKS}_${IDX}" \
    --num_chunks ${CHUNKS} \
    --cache_dir "${CACHE_DIR}" \
    --chunk_idx ${IDX} &
done
wait

# Merge per-chunk outputs
out="${OUT_DIR}/merge.jsonl"; : > "$out"
for IDX in $(seq 0 $((CHUNKS-1))); do
  f_jsonl="${OUT_DIR}/${CHUNKS}_${IDX}.jsonl"
  f_json="${OUT_DIR}/${CHUNKS}_${IDX}.json"
  if   [ -f "$f_jsonl" ]; then cat "$f_jsonl" >> "$out";
  elif [ -f "$f_json"  ]; then cat "$f_json"  >> "$out";
  else echo "WARN: missing ${CHUNKS}_${IDX}.{jsonl|json}"; fi
done
echo "Wrote merged predictions to: $out"

