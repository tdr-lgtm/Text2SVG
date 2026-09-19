import torch
from svgicon.vocab import VocabBridge
from svgicon.model import load_tokenizer, build
from svgicon import config

assert torch.cuda.is_available(), "no GPU"
print(torch.cuda.get_device_name(0))

hf = load_tokenizer()
bridge = VocabBridge(hf, config.MODEL_NAME)
print(bridge.describe())

model, optimizer, info = build(bridge)

print(f"\nresized to        {info['total_vocab']:,}")
print(f"tied embeddings   {info['tied']}")
print(f"lora tensors      {info['lora_tensors']}")
print(f"embedding tensors {info['embedding_tensors']}")
print(f"optimizer         {info['optimizer']}")
print()
model.print_trainable_parameters()

# Verify that the embedding parameter group has weight_decay=0.
# Otherwise, AdamW could shrink the pretrained embedding rows even when their gradients are zero.
for i, group in enumerate(optimizer.param_groups):
    print(f"group {i}: {len(group['params'])} tensors, "
            f"lr {group['lr']}, weight_decay {group['weight_decay']}")

print(f"\nmemory allocated  {torch.cuda.memory_allocated()/1024**3:.1f} GB")