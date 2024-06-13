PROMPT=["A Woman is Blowing Dry Hair"]
OUTDIR="results/1"


BASE_PATH="models/base_t2v/model.ckpt"
CONFIG_PATH="models/base_t2v/model_config.yaml"






python scripts/sample_text2video.py \
      --ckpt_path $BASE_PATH \
      --config_path $CONFIG_PATH \
      --prompt "$PROMPT" \
      --save_dir $OUTDIR \
      --n_samples 500 \
      --batch_size 1 \
      --seed 1000 \
      --show_denoising_progress