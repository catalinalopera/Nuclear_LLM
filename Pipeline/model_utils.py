import torch, os, json, random, bitsandbytes as bnb, torch.nn as nn, psutil
from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, TrainingArguments, BitsAndBytesConfig, Trainer, DataCollatorForLanguageModeling
from trl import SFTTrainer
import re
from pprint import pprint
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, get_peft_model, LoraConfig
import pandas as pd 
from huggingface_hub import login
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

HF_TOKEN = "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"
model_name = "meta-llama/Meta-Llama-3-8B"

class ModelUtils:
    def __init__(self):
        self.tokenizer = None
        self.model = None

    def load_model_and_tokenizer(self):
        try:
            print("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=HF_TOKEN)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "right"

            print("EOS Token:", self.tokenizer.eos_token)
            print("EOS Token ID:", self.tokenizer.eos_token_id)

            # Configure Quantization
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)

            # Load Pretrained Model with Quantization
            print("Loading model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map='auto',
                low_cpu_mem_usage=True,
                use_auth_token=HF_TOKEN
            )

            # Enable Gradient Checkpointing and Prepare for k-bit Training
            print("Applying gradient checkpointing and preparing for k-bit training...")
            self.model.gradient_checkpointing_enable()
            self.model = prepare_model_for_kbit_training(self.model)

            print("Model and tokenizer loaded and configured successfully.")
        except Exception as e:
            print("An error occurred:", e)
            self.model = None
            self.tokenizer = None

    def apply_lora_config(self):
        try:
            if self.model is None:
                raise ValueError("Model is not loaded. Cannot apply LoRA configuration.")

            print("Applying LoRA configuration...")

            # Define LoRA configuration
            lora_config = LoraConfig(
                r=16,
                lora_alpha=16,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM"
            )

            # Apply LoRA configuration to the model
            self.model = get_peft_model(self.model, lora_config)

            print("LoRA configuration applied successfully.")
        except Exception as e:
            print("An error occurred while applying LoRA configuration:", e)
