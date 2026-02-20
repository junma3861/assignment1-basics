# Quick Start Training Script (PowerShell)
# This is a minimal working example that you can run immediately with the provided data

# Ensure you're in the correct directory
cd c:\Users\ma_ju\Desktop\workspace\stanford_cs_336\assignment1-basics

# Small model for quick testing (should run on CPU or small GPU)
python -m cs336_basics.training_together `
    --train_data_path data/TinyStoriesV2-GPT4-train.ids `
    --val_data_path data/TinyStoriesV2-GPT4-valid.ids `
    --vocab_size 10000 `
    --context_length 128 `
    --d_model 256 `
    --num_heads 4 `
    --d_ff 1024 `
    --num_layers 4 `
    --batch_size 32 `
    --lr 5e-4 `
    --min_lr 5e-5 `
    --warmup_iters 500 `
    --max_iters 5000 `
    --checkpoint_interval 1000 `
    --eval_interval 250 `
    --log_interval 50 `
    --checkpoint_dir checkpoints/quickstart `
    --seed 42

Write-Host "Training completed! Check the checkpoints/quickstart directory for saved models."
