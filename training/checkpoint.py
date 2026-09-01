import os
import torch
import glob

def save_checkpoint(model, optimizer, scheduler, step, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    state = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler else None,
        "step": step,
        "rng_state_torch": torch.get_rng_state(),
        "rng_state_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    }
    
    # Save safely by writing to a temp file first then renaming
    temp_path = filepath + ".tmp"
    torch.save(state, temp_path)
    os.replace(temp_path, filepath)
    print(f"Checkpoint saved to {filepath}")

def load_checkpoint(filepath, model, optimizer=None, scheduler=None):
    if not os.path.exists(filepath):
        print(f"No checkpoint found at {filepath}")
        return 0
        
    print(f"Loading checkpoint from {filepath}")
    
    state = torch.load(filepath, map_location='cpu')
    
    model_state = state.get('model_state', state.get('model_state_dict'))
    model.load_state_dict(model_state)
    
    if optimizer:
        opt_state = state.get('optimizer_state', state.get('optimizer_state_dict'))
        if opt_state:
            optimizer.load_state_dict(opt_state)
            
    if scheduler:
        sch_state = state.get('scheduler_state', state.get('scheduler_state_dict'))
        if sch_state:
            scheduler.load_state_dict(sch_state)
            
    if "rng_state_torch" in state:
        torch.set_rng_state(state["rng_state_torch"])
    if "rng_state_cuda" in state and state["rng_state_cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["rng_state_cuda"])
        
    step = state.get('step', 0)
    print(f"Loaded checkpoint from {filepath} (step {step})")
    return step

def get_latest_checkpoint(checkpoint_dir):
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "checkpoint_step_*.pt"))
    if not checkpoints:
        latest = os.path.join(checkpoint_dir, "latest.pt")
        if os.path.exists(latest):
            return latest
        return None
        
    steps = []
    for cp in checkpoints:
        try:
            step = int(cp.split("_step_")[-1].split(".pt")[0])
            steps.append((step, cp))
        except:
            pass
            
    if not steps:
        return None
        
    steps.sort()
    return steps[-1][1]
