import torch
import time
from tqdm import tqdm

from training.checkpoint import save_checkpoint

class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        dataloader,
        device,
        checkpoint_dir="checkpoints",
        gradient_accumulation_steps=1,
        max_grad_norm=1.0,
        save_every_steps=1000,
        mixed_precision=True
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.dataloader = dataloader
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.save_every_steps = save_every_steps
        self.max_grad_norm = max_grad_norm
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.mixed_precision = mixed_precision
        
        # Ensure determinism across python, numpy, and pytorch
        self._set_seed(42)
        
        if self.mixed_precision:
            self.scaler = torch.amp.GradScaler('cuda', enabled=self.device.type == 'cuda')
            
        self.global_step = 0
        
    def _set_seed(self, seed):
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def train(self, num_epochs, start_step=0):
        self.model.train()
        step = start_step
        
        for epoch in range(num_epochs):
            print(f"Starting Epoch {epoch+1}/{num_epochs}")
            pbar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}")
            
            for batch_idx, batch in enumerate(pbar):
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                t0 = time.time()
                
                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16, enabled=self.device.type == 'cuda'):
                    outputs = self.model(input_ids=input_ids, labels=labels)
                    loss = outputs["loss"]
                    loss = loss / self.gradient_accumulation_steps
                
                self.scaler.scale(loss).backward()
                
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    
                    if self.scheduler:
                        self.scheduler.step()
                        
                    step += 1
                    
                    t1 = time.time()
                    dt = t1 - t0
                    tokens_per_sec = (input_ids.numel()) / max(dt, 0.0001)
                    current_lr = self.scheduler.get_last_lr()[0] if self.scheduler else self.optimizer.param_groups[0]['lr']
                    
                    pbar.set_postfix({
                        'loss': f"{loss.item() * self.gradient_accumulation_steps:.4f}",
                        'lr': f"{current_lr:.6f}",
                        'tok/s': f"{tokens_per_sec:.0f}"
                    })
                    
                    if step % self.save_every_steps == 0:
                        save_checkpoint(
                            self.model, 
                            self.optimizer, 
                            self.scheduler, 
                            step, 
                            f"{self.checkpoint_dir}/checkpoint_step_{step}.pt"
                        )
                        save_checkpoint(
                            self.model, 
                            self.optimizer, 
                            self.scheduler, 
                            step, 
                            f"{self.checkpoint_dir}/latest.pt"
                        )
