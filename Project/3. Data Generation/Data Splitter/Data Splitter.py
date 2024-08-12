# %%
import json

# Read the JSON file (with utf-8 encoding)
with open('36744_CNSC_QA_pairs.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Update the keys
updated_data = [{'question': entry['prompt'], 'answer': entry['response']} for entry in data]

# Find the total number of pairs
total_pairs = len(updated_data)
print(f"Total number of pairs: {total_pairs}")

# Split into training and test sets
train_size = int(total_pairs * 0.90)  # 85%
test_size = total_pairs - train_size  # Remaining

train_data = updated_data[:train_size]
test_data = updated_data[train_size:]

# Save the JSON files
with open('train_.json', 'w', encoding='utf-8') as train_file:
    json.dump(train_data, train_file, indent=4, ensure_ascii=False)

with open('test_.json', 'w', encoding='utf-8') as test_file:
    json.dump(test_data, test_file, indent=4, ensure_ascii=False)

print(f"{train_size} pairs saved as train.json.")
print(f"{test_size} pairs saved as test.json.")


# %%
import json

MAX_PAIRS = 10000 

# Read the JSON file (with utf-8 encoding)
try:
    with open('36744_CNSC_QA_pairs.json', 'r', encoding='utf-8') as file:
        data = json.load(file)
except FileNotFoundError:
    print("Error: The file '36744_CNSC_QA_pairs.json' was not found.")
    exit()
except json.JSONDecodeError:
    print("Error: The file '36744_CNSC_QA_pairs.json' is not a valid JSON.")
    exit()

# Update the keys
updated_data = [{'question': entry.get('prompt', ''), 'answer': entry.get('response', '')} for entry in data]

# Limit the total number of pairs
limited_data = updated_data[:MAX_PAIRS]
total_pairs = len(limited_data)
print(f"Total number of pairs (limited): {total_pairs}")

# Split into training and test sets
train_size = int(total_pairs * 0.90)  # 90% for training
test_size = total_pairs - train_size  # Remaining for testing

train_data = limited_data[:train_size]
test_data = limited_data[train_size:]

# Save the JSON files
try:
    with open('train_.json', 'w', encoding='utf-8') as train_file:
        json.dump(train_data, train_file, indent=4, ensure_ascii=False)
    with open('test_.json', 'w', encoding='utf-8') as test_file:
        json.dump(test_data, test_file, indent=4, ensure_ascii=False)
    print(f"{train_size} pairs saved as train_.json.")
    print(f"{test_size} pairs saved as test_.json.")
except IOError as e:
    print(f"Error saving files: {e}")


# %%
def process_answers(data):
    processed_data = []
    for entry in data:
        if isinstance(entry['answer'], list):
            # If 'answer' is a list of options, find the correct answer
            correct_answer = next(item['answer'] for item in entry['answer'] if item['correct'])
            processed_entry = {
                'question': entry['question'],
                'answer': correct_answer
            }
        else:
            # If 'answer' is already a string, keep it as is
            processed_entry = entry
        processed_data.append(processed_entry)
    return processed_data

def process_file(file_path, output_path):
    # Read the JSON file (with utf-8 encoding)
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # Find the total number of question-answer pairs
    total_pairs = len(data)
    print(f"Total number of question-answer pairs in {file_path}: {total_pairs}")

    # Process the answers to ensure 'answer' contains only the correct answer
    processed_data = process_answers(data)

    # Save the processed data to a new JSON file
    with open(output_path, 'w', encoding='utf-8') as processed_file:
        json.dump(processed_data, processed_file, indent=4, ensure_ascii=False)

    print(f"Processed question-answer pairs saved as {output_path}.")

# Process both train.json and test.json
process_file('train.json', 'processed_train.json')
process_file('test.json', 'processed_test.json')


# %%
def merge_files(file_paths, output_path):
    combined_data = []
    
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            combined_data.extend(data)
    
    # Print the total number of question-answer pairs
    total_pairs = len(combined_data)
    print(f"Total number of question-answer pairs in {output_path}: {total_pairs}")
    
    # Save the combined data to the output file
    with open(output_path, 'w', encoding='utf-8') as output_file:
        json.dump(combined_data, output_file, indent=4, ensure_ascii=False)
    
    print(f"Combined data saved as {output_path}.")

# File paths for train data
train_file_paths = ['train_.json', 'processed_train.json']
# File paths for test data
test_file_paths = ['test_.json', 'processed_test.json']

# Merge the files and save the results
merge_files(train_file_paths, 'train.json')
merge_files(test_file_paths, 'test.json')


# %%
train_input_file = '/kaggle/input/baris-fine-tuning/train.json'
train_output_file = '/kaggle/working/train_dataset.json'

test_input_file = '/kaggle/input/baris-fine-tuning/test.json'
test_output_file = '/kaggle/working/test_dataset.json'

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

# %%
train_input_file = '/kaggle/input/baris-fine-tuning/train.json'
train_output_file = '/kaggle/working/train_dataset.json'
test_input_file = '/kaggle/input/baris-fine-tuning/test.json'
test_output_file = '/kaggle/working/test_dataset.json'

def add_questions_key(input_file, output_file):
    with open(input_file, 'r') as infile:
        data = json.load(infile)
    
    formatted_data = {
        "questions": data
    }

    with open(output_file, 'w') as outfile:
        json.dump(formatted_data, outfile, indent=4)

add_questions_key(train_input_file, train_output_file)
add_questions_key(test_input_file, test_output_file)

print("JSON dosyaları formatlandı ve başarıyla kaydedildi.")

with open(train_output_file) as json_file:
    train = json.load(json_file)
with open(test_output_file) as json_file:
    test = json.load(json_file)

pd.DataFrame(train["questions"]).head()
pd.DataFrame(test["questions"]).head()
pprint(train["questions"][0], sort_dicts=False)
pprint(test["questions"][0], sort_dicts=False)

def check_data_format(data):
    if "questions" not in data or not isinstance(data["questions"], list):
        raise ValueError("Veri 'questions' anahtarını içermiyor veya liste değil.")

check_data_format(train)
check_data_format(test)

prompt = """Below is a question paired with an answer. Please write a response that appropriately completes the request.

### Question:
{}

### Answer:
{}"""

special_tokens = tokenizer.special_tokens_map_extended
eos_token = tokenizer.eos_token
eos_token_id = tokenizer.eos_token_id

def formatting_prompts_func(examples):
    questions = examples["question"]
    answers = examples["answer"]
    texts = []
    for question, answer in zip(questions, answers):
        text = prompt.format(question, answer) + eos_token
        texts.append(text)
    return {"text": texts}

def create_and_format_dataset(data):
    dataset_dict = {
        "question": [item["question"] for item in data["questions"]],
        "answer": [item["answer"] for item in data["questions"]],
    }
    dataset = Dataset.from_dict(dataset_dict)
    dataset = dataset.map(formatting_prompts_func, batched=True)
    dataset = dataset.remove_columns(["text"])
    return dataset

train_dataset = create_and_format_dataset(train)
test_dataset = create_and_format_dataset(test)

dataset = DatasetDict({
    'train': train_dataset,
    'test': test_dataset  
})

def preprocess_function(examples):
    inputs = tokenizer(examples['question'], padding='max_length', truncation=True, max_length=128, return_tensors='pt')
    labels = tokenizer(examples['answer'], padding='max_length', truncation=True, max_length=128, return_tensors='pt')
    
    inputs['labels'] = labels['input_ids']
    inputs['attention_mask'] = inputs['attention_mask']
    
    return inputs
dataset['train'] = dataset['train'].map(preprocess_function, batched=True)
dataset['test'] = dataset['test'].map(preprocess_function, batched=True)

print(dataset)
# print(train_dataset[0])
# print(train_dataset)


