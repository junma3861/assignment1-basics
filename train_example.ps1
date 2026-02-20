# Example PowerShell script for training the Transformer Language Model

# Basic training with default hyperparameters
python -m cs336_basics.training_together `
    --train_data_path data/TinyStoriesV2-GPT4-train.ids `
    --val_data_path data/TinyStoriesV2-GPT4-valid.ids `
    --vocab_size 10000 `
    --context_length 256 `
    --d_model 512 `
    --num_heads 8 `
    --d_ff 2048 `
    --num_layers 6 `
    --batch_size 64 `
    --max_iters 10000 `
    --checkpoint_dir checkpoints/run1

# Advanced training with custom hyperparameters and W&B logging
python -m cs336_basics.training_together `
    --train_data_path data/TinyStoriesV2-GPT4-train.ids `
    --val_data_path data/TinyStoriesV2-GPT4-valid.ids `
    --vocab_size 10000 `
    --context_length 512 `
    --d_model 768 `
    --num_heads 12 `
    --d_ff 3072 `
    --num_layers 12 `
    --lr 3e-4 `
    --min_lr 3e-5 `
    --weight_decay 0.1 `
    --beta1 0.9 `
    --beta2 0.95 `
    --warmup_iters 2000 `
    --batch_size 32 `
    --grad_clip 1.0 `
    --max_iters 100000 `
    --checkpoint_interval 5000 `
    --eval_interval 500 `
    --log_interval 100 `
    --checkpoint_dir checkpoints/run2 `
    --use_wandb `
    --wandb_project transformer-lm-cs336 `
    --wandb_run_name large-model-run `
    --seed 42

# Resume training from a checkpoint
python -m cs336_basics.training_together `
    --train_data_path data/TinyStoriesV2-GPT4-train.ids `
    --val_data_path data/TinyStoriesV2-GPT4-valid.ids `
    --vocab_size 10000 `
    --context_length 256 `
    --d_model 512 `
    --num_heads 8 `
    --d_ff 2048 `
    --num_layers 6 `
    --checkpoint_dir checkpoints/run1 `
    --resume_from checkpoints/run1/checkpoint_iter_5000.pt `
    --max_iters 20000
