import sys
import os
import argparse
import importlib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from tokenizer.tokenizer import GenPyTokenizer
from training.instruction_dataset import InstructionDataset
from training.optimizer import create_optimizer
from training.scheduler import get_cosine_schedule_with_warmup
from training.trainer import Trainer
from training.checkpoint import load_checkpoint, get_latest_checkpoint
from model.genpy_llm import GenPyLLM, count_parameters

def main():
    parser = argparse.ArgumentParser(description="Fine-tune GenPy-LLM on instruction data")
    parser.add_argument("--config", type=str, required=True, help="Config file (e.g. configs/tiny.py)")
    parser.add_argument("--data", type=str, required=True, help="Path to instruction JSONL file")
    parser.add_argument("--tokenizer", type=str, default="tokenizer/genpy_tokenizer.json")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to pretrained checkpoint to start from")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (lower than pretraining)")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/sft")
    parser.add_argument("--save-every", type=int, default=100)
    args = parser.parse_args()
    
    config_module_path = args.config.replace('/', '.').replace('\\', '.').replace('.py', '')
    config_module = importlib.import_module(config_module_path)
    config = config_module.config
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    tokenizer = GenPyTokenizer(args.tokenizer)
    
    print("Initializing model...")
    model = GenPyLLM(config).to(device)
    
    start_step = 0
    if args.checkpoint and os.path.exists(args.checkpoint):
        start_step = load_checkpoint(args.checkpoint, model)
        print(f"Loaded checkpoint from {args.checkpoint}, resuming from step {start_step}")
    elif not args.checkpoint:
        latest = get_latest_checkpoint("checkpoints")
        if latest:
            start_step = load_checkpoint(latest, model)
            print(f"Loaded latest checkpoint from {latest}")
    
    count_parameters(model)
    
    dataset = InstructionDataset(args.data, tokenizer, config.max_seq_len)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    optimizer = create_optimizer(model, learning_rate=args.lr)
    total_steps = len(dataloader) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=max(1, total_steps // 20), num_training_steps=total_steps)
    
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
    print("Fine-tuning complete.")

if __name__ == "__main__":
    main()
