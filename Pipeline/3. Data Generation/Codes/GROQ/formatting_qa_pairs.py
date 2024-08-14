import os
import json

def load_qa_pairs(input_file_path):
    """
    Loads QA pairs from a JSON file.

    Parameters:
        input_file_path (str): The file path to the input JSON file.

    Returns:
        list: A list of QA pairs.
    """
    input_file_path = input_file_path.strip('"')
    with open(input_file_path, 'r', encoding='utf-8') as file:
        qa_pairs = json.load(file)
    print(f"Loaded {len(qa_pairs)} QA pairs from {input_file_path}")
    return qa_pairs

def format_qa_pairs(qa_pairs):
    """
    Formats QA pairs for the Llama 3 8B quantized model.

    Parameters:
        qa_pairs (list): A list of QA pairs.

    Returns:
        list: A list of formatted QA pairs.
    """
    formatted_pairs = []
    
    for pair in qa_pairs:
        formatted_pairs.append({
            "prompt": pair.get("question", ""),
            "response": pair.get("answer", "")
        })
    
    print(f"Formatted {len(formatted_pairs)} QA pairs")
    return formatted_pairs

def save_formatted_qa_pairs(formatted_pairs, output_file_path):
    """
    Saves formatted QA pairs to a JSON file.

    Parameters:
        formatted_pairs (list): A list of formatted QA pairs.
        output_file_path (str): The file path to save the JSON file.
    """
    output_file_path = output_file_path.strip('"')
    with open(output_file_path, 'w', encoding='utf-8') as file:
        json.dump(formatted_pairs, file, ensure_ascii=False, indent=4)
    print(f"Saved formatted QA pairs to {output_file_path}")

def main():
    input_file_path = input("Enter the path to the input JSON file: ").strip()
    output_file_name = "formatted_qa_pairs.json"
    output_dir = input("Enter the directory to save the formatted JSON file: ").strip()
    output_file_path = os.path.join(output_dir.strip('"'), output_file_name)

    qa_pairs = load_qa_pairs(input_file_path)
    formatted_pairs = format_qa_pairs(qa_pairs)
    save_formatted_qa_pairs(formatted_pairs, output_file_path)

if __name__ == "__main__":
    main()
