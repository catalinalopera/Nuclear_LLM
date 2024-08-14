import json
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

class DataProcessor:
    def __init__(self, tokenizer_name, eos_token, max_seq_length):
        # Initialize tokenizer and special tokens
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.eos_token = eos_token
        self.max_seq_length = max_seq_length
    
    def add_questions_key(self, input_file, output_file):
        """Read the JSON file and add the 'questions' key."""
        with open(input_file, 'r') as infile:
            data = json.load(infile)        
        formatted_data = {"questions": data}
        with open(output_file, 'w') as outfile:
            json.dump(formatted_data, outfile, indent=4)
        print(f"Formatted data saved to {output_file}")

    def check_data_format(self, data):
        """Check if 'questions' key exists and is a list."""
        if "questions" not in data or not isinstance(data["questions"], list):
            raise ValueError("The data does not contain the 'questions' key or it is not a list.")

    def formatting_prompts_func(self, examples):
        """Format the prompts for the dataset."""
        questions = examples["question"]
        answers = examples["answer"]
        texts = [f"Below is a question paired with an answer. Write a response that appropriately completes the request.\n\n### Question:\n{q}\n\n### Answer:\n{a}{self.eos_token}" for q, a in zip(questions, answers)]
        return {"text": texts}

    def create_and_format_dataset(self, data):
        """Create and format a dataset from the data."""
        dataset_dict = {
            "question": [item["question"] for item in data["questions"]],
            "answer": [item["answer"] for item in data["questions"]],
        }
        dataset = Dataset.from_dict(dataset_dict)
        dataset = dataset.map(self.formatting_prompts_func, batched=True)
        dataset = dataset.remove_columns(["text"])
        return dataset

    def preprocess_function(self, examples):
        """Tokenize the input texts."""
        inputs = self.tokenizer(examples['question'], padding='max_length', truncation=True, max_length=self.max_seq_length)
        labels = self.tokenizer(examples['answer'], padding='max_length', truncation=True, max_length=self.max_seq_length)
        inputs['labels'] = labels['input_ids']
        inputs['attention_mask'] = inputs['attention_mask']
        return inputs

    def load_and_prepare_data(self, train_input_file, train_output_file, test_input_file, test_output_file):
        """Load, format, and preprocess datasets."""
        self.add_questions_key(train_input_file, train_output_file)
        self.add_questions_key(test_input_file, test_output_file)

        with open(train_output_file) as json_file:
            train = json.load(json_file)
        with open(test_output_file) as json_file:
            test = json.load(json_file)
        
        self.check_data_format(train)
        self.check_data_format(test)

        train_dataset = self.create_and_format_dataset(train)
        test_dataset = self.create_and_format_dataset(test)

        train_dataset = train_dataset.map(self.preprocess_function, batched=True)
        test_dataset = test_dataset.map(self.preprocess_function, batched=True)

        dataset_dict = DatasetDict({
            'train': train_dataset,
            'test': test_dataset
        })

        print(dataset_dict)
        return dataset_dict

# Parameters
tokenizer_name = "meta-llama/Meta-Llama-3-8B"
eos_token = "<eos>"  # Adjust based on your tokenizer
max_seq_length = 512

# Initialize DataProcessor and load data
data_processor = DataProcessor(tokenizer_name, eos_token, max_seq_length)
dataset_dict = data_processor.load_and_prepare_data(
    train_input_file='train.json',
    train_output_file='train_dataset.json',
    test_input_file='test.json',
    test_output_file='test_dataset.json'
)
