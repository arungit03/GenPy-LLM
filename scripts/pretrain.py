import sys
import os
import argparse
import importlib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from tokenizer.tokenizer import GenPyTokenizer
from training.dataset import create_dataloader
from training.optimizer import create_optimizer
from training.scheduler import get_cosine_schedule_with_warmup
from training.trainer import Trainer
from training.checkpoint import load_checkpoint, get_latest_checkpoint
from model.genpy_llm import GenPyLLM, count_parameters

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file (e.g., configs/tiny.py)")
    parser.add_argument("--data", type=str, default="data/raw/sample.jsonl", help="Path to training data")
    parser.add_argument("--tokenizer", type=str, default="tokenizer/genpy_tokenizer.json", help="Path to tokenizer")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--save-every", type=int, default=10, help="Save checkpoint every N steps")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    
    args = parser.parse_args()
    
    config_module_path = args.config.replace('/', '.').replace('\\', '.').replace('.py', '')
    config_module = importlib.import_module(config_module_path)
    config = config_module.config
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        
    tokenizer = GenPyTokenizer(args.tokenizer)
    if tokenizer.vocab_size == 0:
        print("Tokenizer not found! Run scripts/train_tokenizer.py first.")
        return
        
    print("Initializing model...")
    model = GenPyLLM(config).to(device)
    count_parameters(model)
    
    dataloader = create_dataloader(
        args.data, 
        tokenizer, 
        config.max_seq_len, 
        args.batch_size, 
        shuffle=True
    )
    
    optimizer = create_optimizer(model, learning_rate=3e-4)
    
    total_steps = len(dataloader) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=max(1, total_steps // 10), num_training_steps=total_steps)
    
    start_step = 0
    if args.resume:
        latest = get_latest_checkpoint(args.checkpoint_dir)
        if latest:
            start_step = load_checkpoint(latest, model, optimizer, scheduler)
            
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        dataloader=dataloader,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        save_every_steps=args.save_every
    )
    
    trainer.train(num_epochs=args.epochs, start_step=start_step)

if __name__ == "__main__":
    main()
