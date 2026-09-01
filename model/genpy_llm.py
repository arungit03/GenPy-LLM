import torch
import torch.nn as nn

from model.config import ModelConfig
from model.transformer import Transformer

class GenPyLLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.transformer = Transformer(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        if config.tie_embeddings:
            self.lm_head.weight = self.transformer.tok_embeddings.weight
            
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor = None, attention_mask: torch.Tensor = None):
        hidden_states = self.transformer(input_ids, attention_mask=attention_mask)
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            # Our dataset already provides shifted labels where labels[i] is the target for input_ids[i]
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))
            
        return {"logits": logits, "loss": loss}

    @classmethod
    def from_checkpoint(cls, path):
        checkpoint = torch.load(path, map_location="cpu")
        # Ensure config matches what was saved
        config_dict = checkpoint["config"]
        # Convert dictionary to config object if needed
        if isinstance(config_dict, dict):
            config = ModelConfig(**config_dict)
        else:
            config = config_dict
            
        model = cls(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        return model

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    print("GenPy-LLM")
    print("-" * 9)
    print(f"Parameters: {total_params:,}")
    print(f"Trainable: {trainable_params:,}")
    print(f"Non-trainable: {non_trainable_params:,}")
    
    components = {}
    for name, param in model.named_parameters():
        component = name.split('.')[0] if '.' in name else name
        if component == "transformer":
            sub = name.split('.')[1] if len(name.split('.')) > 1 else ""
            component = f"transformer.{sub}"
        components[component] = components.get(component, 0) + param.numel()
        
    print("\nParameters by component:")
    for component, count in components.items():
        print(f"  {component}: {count:,}")
    
    return total_params, trainable_params

