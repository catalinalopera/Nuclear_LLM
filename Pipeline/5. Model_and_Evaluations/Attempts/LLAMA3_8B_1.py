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
    device_map='auto',  # Otomatik olarak CPU ve GPU'ya dağıtma
    low_cpu_mem_usage=True,
    token=HF_TOKEN,

)     # max_length=max_seq_length,

# %%
# Enter the sentence
text = "Hello, how are you?",
inputs = tokenizer(text, return_tensors="pt").to('cuda')

# Model testing
outputs = model.generate(**inputs)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(generated_text)

# %%
# Checking model quantization
def is_8bit_quantized(model):
    for name, param in model.named_parameters():
        if param.dtype == torch.int8:
            print(f"Parameter {name} is quantized to 8-bit.")
        else:
            print(f"Parameter {name} is NOT quantized to 8-bit.")

is_8bit_quantized(model)

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

# %%
lora_config = {
    "r": 16,  # Number of LoRA layers
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",  # Target modules for LoRA
                       "gate_proj", "up_proj", "down_proj"],
    "lora_alpha": 16,  # Alpha value for LoRA (optional)
    "lora_dropout": 0.1,  # Dropout value for LoRA (optional)
    "bias": "none",  # Type of bias for LoRA (optional)
    "use_gradient_checkpointing": True,  # Use of gradient checkpointing
    "use_rslora": False,  # Use of RSLora (optional)
    "use_dora": False,  # Use of DoRa (optional)
    "loftq_config": None  # Configuration for LoFTQ (optional)
}


# %%
# Training configuration
training_config = {
    "per_device_train_batch_size": 2,        # Batch size per device
    "gradient_accumulation_steps": 4,        # Gradient accumulation steps
    "warmup_steps": 5,                       # Warmup steps
    "max_steps": 0,                          # Maximum steps (0 if epochs are defined)
    "num_train_epochs": 10,                  # Number of training epochs (0 if maximum steps are defined)
    "learning_rate": 2e-4,                   # Learning rate
    "fp16": not torch.cuda.is_bf16_supported(),  # Use fp16 if bf16 is not supported
    "bf16": torch.cuda.is_bf16_supported(),  # Use bf16 if supported
    "logging_steps": 1,                      # Logging steps
    "optim": "adamw",                        # Optimizer
    "weight_decay": 0.01,                    # Weight decay
    "lr_scheduler_type": "linear",           # Learning rate scheduler
    "seed": 42,                              # Seed value
    "output_dir": "outputs",                 # Output directory
}


# %%


# Load the model and tokenizer
def load_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer

# Inference logic to generate responses from the model
def generate_response(model, tokenizer, question, max_length=50, temperature=0.7, top_k=50):
    inputs = tokenizer.encode(question, return_tensors='pt')
    outputs = model.generate(
        inputs,
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
        no_repeat_ngram_size=2
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Function to evaluate accuracy
def evaluate_accuracy(model, tokenizer, questions, answers):
    correct = 0
    for question, answer in zip(questions, answers):
        response = generate_response(model, tokenizer, question)
        if response.strip().lower() == answer.strip().lower():
            correct += 1
    accuracy = correct / len(questions)
    return accuracy

# Function to evaluate truthfulness
def evaluate_truthfulness(model, tokenizer, questions, truth_checker):
    truthful_responses = 0
    for question in questions:
        response = generate_response(model, tokenizer, question)
        if truth_checker(response):
            truthful_responses += 1
    truthfulness = truthful_responses / len(questions)
    return truthfulness

# Function to measure latency and memory usage
def measure_latency_memory(model, tokenizer, questions):
    latencies = []
    for question in questions:
        start_time = time()
        _ = generate_response(model, tokenizer, question)
        latency = time() - start_time
        latencies.append(latency)
    avg_latency = np.mean(latencies)
    memory_usage = psutil.Process().memory_info().rss / (1024 ** 2)  # Memory usage in MB
    return avg_latency, memory_usage

# Function to evaluate hallucinations
def evaluate_hallucinations(model, tokenizer, questions, fact_checker):
    hallucination_count = 0
    for question in questions:
        response = generate_response(model, tokenizer, question)
        if not fact_checker(response):
            hallucination_count += 1
    hallucination_rate = hallucination_count / len(questions)
    return hallucination_rate

# Placeholder functions for truth and fact checkers
def truth_checker(response):
    #
    return True

def fact_checker(response):
    #
    return True

# Example data
questions = ["What is the capital of France?", "Who wrote '1984'?", "What is 2+2?"]
answers = ["Paris", "George Orwell", "4"]

# Load models
model_name = "ll"
model, tokenizer = load_model_and_tokenizer(model_name)

# Evaluating the model
accuracy = evaluate_accuracy(model, tokenizer, questions, answers)
truthfulness = evaluate_truthfulness(model, tokenizer, questions, truth_checker)
avg_latency, memory_usage = measure_latency_memory(model, tokenizer, questions)
hallucination_rate = evaluate_hallucinations(model, tokenizer, questions, fact_checker)

# Printing the results
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Truthfulness: {truthfulness * 100:.2f}%")
print(f"Average Latency: {avg_latency:.4f} seconds")
print(f"Memory Usage: {memory_usage:.2f} MB")
print(f"Hallucination Rate: {hallucination_rate * 100:.2f}%")



