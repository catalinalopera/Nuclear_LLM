# %% [markdown]
# # 1. GitHub Clone

# %%
!git clone https://github.com/Falgun1/NLP-Corpus
%cd NLP-Corpus/Pipeline

# %% [markdown]
# # 2. Library

# %%
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
assert device == torch.device('cuda'), "Not using CUDA. Set: Runtime > Change runtime type > Hardware Accelerator: GPU"

# %%
%%capture
!pip install -q bitsandbytes
!pip install -q transformers
!pip install -q nltk
!pip install -q datasets
!pip install -q textstat
!pip install -q rouge_score
major_version, minor_version = torch.cuda.get_device_capability()
if major_version >= 8:
    !pip install --no-deps packaging ninja einops flash-attn xformers trl peft accelerate bitsandbytes
else:
    !pip install --no-deps xformers trl peft accelerate bitsandbytes
pass

# %%
import torch,os, json, re, random  
import bitsandbytes as bnb
import torch.nn as nn
import pandas as pd
from pprint import pprint
from datasets import Dataset, DatasetDict, load_dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, Trainer, TrainingArguments, DataCollatorForLanguageModeling)
from peft import prepare_model_for_kbit_training, get_peft_model, LoraConfig
from huggingface_hub import login
from trl import SFTTrainer
from keywords_manager import KeywordsManager
from wiki import WikiArticleFetcher, FilteredWikiArticleFetcher
from file_utils import ZipExtractor
from generator import QuestionGenerator, print_qa
from question_generator import QuestionAnswerGenerator

# %% [markdown]
# # 3.Web Scraping

# %%
def data_collector():
    wscraping = FilteredWikiArticleFetcher(keywords_manager=KeywordsManager(),file_limit=5,filtered_names = ['wiki_CNSC'] )
    wscraping.fetch_and_save_articles()  
if __name__ == "__main__":
    data_collector()

# %%
def zip_extractor():
    extractor = ZipExtractor(zip_path = 'filtered_articles.zip', extract_to = 'Articles')
    extractor.extract()
if __name__ == "__main__":
    zip_extractor()  
def list_files_in_directory(directory):
    if not os.path.exists(directory):
        print(f"The directory {directory} does not exist.")
        return []
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    return files
def print_files():
    directory = 'Articles'
    files = list_files_in_directory(directory)
    if files:
        print(f"Files in '{directory}' directory:")
        for file in files:
            print(file)
    else:
        print("No files found.")
if __name__ == "__main__":
    print_files()

# %% [markdown]
# # 4.Q&A Generator

# %%
def main():  
    qag = QuestionAnswerGenerator(articles_folder = "Articles" , num_questions = 20, answer_style = 'all')
    qag.generate_questions()
if __name__ == "__main__":
    main()

# %%
# def read_json_file(file_path):
#     with open(file_path, 'r', encoding='utf-8') as file:
#         data = json.load(file)
#     return data

# def print_json_content(file_path):
#     data = read_json_file(file_path)
#     print(f"Content of {file_path}:")
#     print(json.dumps(data, indent=4))  # Pretty-print the JSON data

# train_file_path = 'train.json'
# test_file_path = 'test.json'

# print_json_content(train_file_path)
# print_json_content(test_file_path)

# import json
# def count_records(file_path):
#     with open(file_path, 'r') as file:
#         data = json.load(file)
#     return len(data)
# def main():
#     train_file_path = 'train.json'
#     test_file_path = 'test.json' 
#     train_count = count_records(train_file_path)
#     test_count = count_records(test_file_path)  
#     print(f"Number of records in train.json: {train_count}")
#     print(f"Number of records in test.json: {test_count}")
# if __name__ == "__main__":
#     main()

# %% [markdown]
# # 5. Model Loader

# %%
HF_TOKEN = "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"
model_name = "meta-llama/Meta-Llama-3-8B"
max_seq_length = 2048

def load_model_and_tokenizer():
    """Load the model and tokenizer with configurations."""
    try:
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=HF_TOKEN)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        special_tokens = tokenizer.special_tokens_map_extended
        eos_token = tokenizer.eos_token
        eos_token_id = tokenizer.eos_token_id
        
        print("EOS Token:", eos_token)
        print("EOS Token ID:", eos_token_id)
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        print("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map='auto',  
            low_cpu_mem_usage=True,
            use_auth_token=HF_TOKEN
        )
        print("Applying gradient checkpointing and preparing for k-bit training...")
        model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model)
        print("Model and tokenizer loaded and configured successfully.")
        return model, tokenizer
    except Exception as e:
        print("An error occurred:", e)
        return None, None

def apply_lora_config(model):
    """Apply LoRA configuration to the model."""
    try:
        print("Applying LoRA configuration...")
        lora_config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)
        print("LoRA configuration applied successfully.")
        return model
    except Exception as e:
        print("An error occurred while applying LoRA configuration:", e)
        return model
model, tokenizer = load_model_and_tokenizer()

if model and tokenizer:
    model = apply_lora_config(model)

# %% [markdown]
# # 6. Dataset Converting to Appropriate Format for Huggingface Transformers

# %%
def add_questions_key(input_file, output_file):
    # Read the JSON file
    with open(input_file, 'r') as infile:
        data = json.load(infile)        
    # Format the data by adding the 'questions' key
    formatted_data = {
        "questions": data
    }

    # Write the formatted data to a new file
    with open(output_file, 'w') as outfile:
        json.dump(formatted_data, outfile, indent=4)
# File paths for JSON datasets
train_input_file = 'train.json'
train_output_file = 'train_dataset.json'

test_input_file = 'test.json'
test_output_file = 'test_dataset.json'

# Convert training and test datasets to the appropriate format
add_questions_key(train_input_file, train_output_file)
add_questions_key(test_input_file, test_output_file)

print("JSON files have been formatted and saved successfully.")

with open("test_dataset.json") as json_file:
    test = json.load(json_file)    
with open("train_dataset.json") as json_file:
    train = json.load(json_file)
pd.DataFrame(train["questions"]).head()
pd.DataFrame(test["questions"]).head()
pprint(train["questions"][0], sort_dicts=False)
pprint(test["questions"][0], sort_dicts=False)

# %%
# Function to check data format
def check_data_format(data):
    if "questions" not in data or not isinstance(data["questions"], list):
        raise ValueError("The data does not contain the 'questions' key or it is not a list.")

check_data_format(train)
check_data_format(test)

# Define the prompt format
prompt = """Below is a question paired with an answer. Please write a response that appropriately completes the request.

### Question:
{}

### Answer:
{}"""

# Get special tokens and EOS token from the tokenizer
special_tokens = tokenizer.special_tokens_map_extended
eos_token = tokenizer.eos_token
eos_token_id = tokenizer.eos_token_id

# Function to format prompts
def formatting_prompts_func(examples):
    questions = examples["question"]
    answers = examples["answer"]
    texts = []
    for question, answer in zip(questions, answers):
        # Format the text according to the prompt and append eos_token
        text = prompt.format(question, answer) + eos_token
        texts.append(text)
    return {"text": texts}

# Function to convert data into dataset format
def create_and_format_dataset(data):
    dataset_dict = {
        "question": [item["question"] for item in data["questions"]],
        "answer": [item["answer"] for item in data["questions"]],
    }
    dataset = Dataset.from_dict(dataset_dict)
    # Apply the formatting prompts function and remove 'text' column
    dataset = dataset.map(formatting_prompts_func, batched=True)
    dataset = dataset.remove_columns(["text"])
    return dataset

# Create and format training and test datasets
train_dataset = create_and_format_dataset(train)
test_dataset = create_and_format_dataset(test)

# Create a DatasetDict
dataset_dict = DatasetDict({
    'train': train_dataset,
    'test': test_dataset
})
# Check the formatted dataset
print(dataset_dict)

# %%
def preprocess_function(examples):
    # Tokenize the input texts
    inputs = tokenizer(examples['question'], padding='max_length', truncation=True, max_length=max_seq_length, return_tensors='pt')
    labels = tokenizer(examples['answer'], padding='max_length', truncation=True, max_length=max_seq_length, return_tensors='pt')
    
    # Add labels to inputs
    inputs['labels'] = labels['input_ids']
    
    # Create attention masks for the inputs
    inputs['attention_mask'] = inputs['attention_mask']
    
    return inputs

# Apply preprocessing
train_dataset = train_dataset.map(preprocess_function, batched=True)
test_dataset = test_dataset.map(preprocess_function, batched=True)
# print(train_dataset[0])
# print(train_dataset)

# %% [markdown]
# # 6. Training

# %%
login(token="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")
OUTPUT_DIR = "experiments"

training_args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    learning_rate=2e-4,
    fp16=True,
    save_total_limit=3,
    logging_steps=10,
    output_dir=OUTPUT_DIR,
    max_steps=5,
    optim="paged_adamw_8bit",
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    report_to="tensorboard",
    evaluation_strategy="steps",
    eval_steps=10,
    save_strategy="steps",
    save_steps=10
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

model.eval()

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)
model.config.use_cache = False
trainer.train()

# %% [markdown]
# # 7. Model Save and Load

# %%
# def save_model_and_tokenizer(output_dir, model, tokenizer):
#     model.save_pretrained(output_dir)
#     tokenizer.save_pretrained(output_dir)
#     print(f"Model and tokenizer saved to {output_dir}")

# # Model and tokenizer save
# save_model_and_tokenizer(OUTPUT_DIR, model, tokenizer)

# def load_model_and_tokenizer(output_dir):
#     model = AutoModelForCausalLM.from_pretrained(output_dir)
#     tokenizer = AutoTokenizer.from_pretrained(output_dir)
#     print(f"Model and tokenizer loaded from {output_dir}")
#     return model, tokenizer
# loaded_model, loaded_tokenizer = load_model_and_tokenizer(OUTPUT_DIR)
# # Model evaluation mode
# loaded_model.eval()

# %%
trainer.save_model()

# %%
# Define the prompt format
prompt = """Below is a question paired with an answer. Please write a response that appropriately completes the request.

### Question:
{}

### Answer:
{}"""

def generate_answer(question):
    # Format the prompt with the question
    formatted_prompt = prompt.format(question, "")

    # Tokenize the formatted prompt
    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_length=2048,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id
        )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = answer.split('### Answer:')[-1].strip()
    return answer

# %%
test_questions = [
    "What is the CNSC"
]

for question in test_questions:
    print(f"Question: {question}")
    print(f"Answer: {generate_answer(question)}\n")

# %%
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from datasets import load_dataset, load_metric
from textstat.textstat import textstatistics

# Load metric
rouge = load_metric("rouge")

# Load model and tokenizer
model_name = "meta-llama/Meta-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained("experiments")  

# Evaluate ROUGE scores
def evaluate_rouge(predictions, references):
    results = rouge.compute(predictions=predictions, references=references)
    return results

# Calculate readability complexity
def calculate_readability(text):
    complexity = textstatistics().flesch_reading_ease(text)
    return complexity

# Load TrueQA dataset
truthfulqa = load_dataset("truthfulqa")

# Evaluation function for TrueQA
def evaluate_truthfulqa(model, tokenizer, dataset):
    scores = []
    for item in dataset:
        question = item["question"]
        reference_answer = item["answer"]
        generated_answer = generate_answer(question, model, tokenizer)
        
        # Evaluate using ROUGE
        rouge_result = evaluate_rouge([generated_answer], [reference_answer])
        scores.append(rouge_result)
    return scores

# Evaluate on TrueQA validation dataset
truthfulqa_scores = evaluate_truthfulqa(model, tokenizer, truthfulqa["validation"])
print(truthfulqa_scores)

# MLM pipeline (Optional: If model supports fill-mask)
mlm_pipeline = pipeline("fill-mask", model=model, tokenizer=tokenizer)
masked_sentence = "The capital of [MASK] is Paris."
results = mlm_pipeline(masked_sentence)
print(results)



