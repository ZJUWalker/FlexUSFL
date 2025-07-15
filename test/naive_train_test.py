import os
import torch
import torch.nn as nn
import torch.optim as optim

from usfl.llm.split_config import SplitModelConfig
from usfl.utils import load_utils
from transformers import AutoModelForCausalLM, AutoTokenizer


# load server model
model_name = 'meta-llama/llama3.2-1b'
model_dir = os.path.join("/share/models", model_name)
model = AutoModelForCausalLM.from_pretrained(model_dir)
# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_dir)
tokenizer.pad_token = tokenizer.eos_token
# load dataset
device = 'cuda:0'
print(f'cuda memory allocated: {torch.cuda.memory_allocated(device)}')
head, tail = load_utils.load_llama_client(model, SplitModelConfig(5, -1, 5))
head.to(device)
tail.to(device)
print(f'after model loaded, cuda memory allocated: {torch.cuda.memory_allocated(device)/1024**3:.2f} GB')
optimizer = optim.Adam(list(head.parameters()) + list(tail.parameters()), lr=1e-4)
print(f'after optimizer created, cuda memory allocated: {torch.cuda.memory_allocated(device)/1024**3:.2f} GB')
dataloaders = load_utils.load_dataset('gsm8k', tokenizer, [0])
for idx, batch in enumerate(dataloaders[0]['train']):
    if idx > 10:
        break
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch["labels"].to(device) if "labels" in batch else input_ids
    head_outputs = head(input_ids=input_ids, attention_mask=attention_mask)
    tail_outputs = tail.forward(
        hidden_states=head_outputs[0],
        attention_mask=head_outputs[1],
        position_embeddings=head_outputs[2],
        labels=labels,
    )
    torch.cuda.synchronize()
    print(
        f'after forward pass {idx},input size: {input_ids.shape}, cuda memory allocated: {torch.cuda.memory_allocated(device)/1024**3:.2f} GB,max memory allocated: {torch.cuda.max_memory_allocated(device)/1024**3:.2f} GB'
    )
    if tail_outputs.loss is not None:
        optimizer.zero_grad()
        tail_outputs.loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        print(
            f'after backward pass {idx}, cuda memory allocated: {torch.cuda.memory_allocated(device)/1024**3:.2f} GB,max memory allocated: {torch.cuda.max_memory_allocated(device)/1024**3:.2f} GB'
        )
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
