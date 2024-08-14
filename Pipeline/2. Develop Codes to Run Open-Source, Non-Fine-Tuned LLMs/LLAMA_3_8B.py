# %% [markdown]
# # LLAMA3
# https://ai.meta.com/blog/meta-llama-3/
# 
# https://github.com/meta-llama/llama3/blob/main/MODEL_CARD.md

# %% [markdown]
# # 1. Connecting to Google Drive and Changing Directory

# %%
import shutil, os, subprocess
from google.colab import drive
drive.mount('/content/drive')
os.chdir('/content/drive/MyDrive/Colab Notebooks/')

# %%
! python --version

# %% [markdown]
# # 2. GPU detection to prevent version conflicts

# %%
%%capture
import torch
!pip install bitsandbytes
!pip install datasets
major_version, minor_version = torch.cuda.get_device_capability()
if major_version >= 8:
    !pip install --no-deps packaging ninja einops flash-attn xformers trl peft accelerate bitsandbytes
else:
    !pip install --no-deps xformers trl peft accelerate bitsandbytes
pass

# %%
!pip list | grep transformers
!pip list | grep torch
!pip list | grep accelerate
!pip list | grep bitsandbytes
!pip list | grep peft
!pip list | grep trl

# %% [markdown]
# # 3. Import Python Packages

# %%
import torch, os, json, random, bitsandbytes as bnb, torch.nn as nn, psutil
from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, TrainingArguments
from peft import get_peft_model, LoraConfig # LoraConfig: Configuration for LoRA (Low-Rank Adaptation), a technique for parameter-efficient training.


# %% [markdown]
# # 4. Login to Hugging Face

# %%
from huggingface_hub import notebook_login
notebook_login() #TOKEN IS "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"

# %% [markdown]
# # 5. LLAMA 3 8B 8bit quantized

# %%
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
HF_TOKEN = "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"
model_name = "meta-llama/Meta-Llama-3-8B"
max_seq_length = 2048
tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)  # max_length=max_seq_length

# %%
special_tokens = tokenizer.special_tokens_map_extended
eos_token = tokenizer.eos_token
eos_token_id = tokenizer.eos_token_id

print("EOS Token:", eos_token)
print("EOS Token ID:", eos_token_id)

# %%
special_tokens = tokenizer.special_tokens_map_extended
eos_token = tokenizer.eos_token
eos_token_id = tokenizer.eos_token_id

print("EOS Token:", eos_token)
print("EOS Token ID:", eos_token_id)

# %%
quantization_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map='auto',  
    low_cpu_mem_usage=True,
    token=HF_TOKEN,

)     # max_length=max_seq_length,

# %%
# Enter the sentence
text = "Hello, how are you?"
inputs = tokenizer(text, return_tensors="pt").to('cuda')

# Model testing
outputs = model.generate(**inputs)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(generated_text)

# %%
print(f"tokenizer memory usage: {psutil.virtual_memory().used / 1e9} GB")
print(f"Memory usage: {psutil.virtual_memory().used / 1e9} GB")
print("Memory usage summary after model setup:")
print(torch.cuda.memory_summary())

# %%
total_params = sum(p.numel() for p in model.parameters())
print(f"total parameter: {total_params}")

# %%
def model_size(model):
    total_size = 0
    for name, param in model.named_parameters():
        total_size += param.numel() * param.element_size()
    return total_size / (1024**2)  # MB

print(f"Dimension of Model: {model_size(model):.2f} MB")


