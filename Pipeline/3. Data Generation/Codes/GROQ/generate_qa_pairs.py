import os
import json
from groq import Groq

def extract_text_from_file(file_path):
    """
    Extracts text from a text file.

    Parameters:
        file_path (str): The file path to the text file.

    Returns:
        str: The extracted text from the file.
    """
    file_path = file_path.strip('"')  # Remove extra quotes if present
    file_path = os.path.abspath(file_path)  # Normalize path
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The file at {file_path} does not exist.")
    
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

def chunk_text_by_characters(text, chunk_size=2000):
    """
    Chunk the text into smaller pieces based on character size.

    Parameters:
        text (str): The text to be chunked.
        chunk_size (int): The size of each chunk in characters.

    Returns:
        list: A list of text chunks.
    """
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def chunk_text_by_words(text, chunk_size=2000):
    """
    Chunk the text into smaller pieces based on word count.

    Parameters:
        text (str): The text to be chunked.
        chunk_size (int): The size of each chunk in words.

    Returns:
        list: A list of text chunks.
    """
    words = text.split()
    return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

def generate_qa_pairs(text_chunk, num_pairs, api_key):
    """
    Generates QA pairs from the given text chunk using the Groq API.

    Parameters:
        text_chunk (str): The input text chunk for generating QA pairs.
        num_pairs (int): The number of QA pairs to generate.
        api_key (str): The API key for accessing the Groq API.

    Returns:
        list: A list of QA pairs for the text chunk.
    """
    client = Groq(api_key=api_key)
    
    prompt = "Generate QA pairs based on the following document:\n\n" + text_chunk
    
    try:
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="llama3-8b-8192"
        )
        
        response_content = response.choices[0].message.content
        lines = [line.strip() for line in response_content.split('\n') if line.strip()]

        qa_pairs = []
        # Generate QA pairs based on the structure of the output
        for i in range(len(lines)):
            if i % 2 == 0:  # Even index lines are questions
                qa_pairs.append({
                    'prompt': lines[i],
                    'response': lines[i + 1] if i + 1 < len(lines) else ''
                })
        
        return qa_pairs[:num_pairs]
    except Exception as e:
        print(f"Error generating QA pairs: {e}")
        return []

def save_qa_pairs_to_json(qa_pairs, output_file_path):
    """
    Saves QA pairs to a JSON file.

    Parameters:
        qa_pairs (list): A list of QA pairs.
        output_file_path (str): The file path to save the JSON file.
    """
    try:
        with open(output_file_path, 'w', encoding='utf-8') as file:
            json.dump(qa_pairs, file, ensure_ascii=False, indent=4)
    except IOError as e:
        print(f"Error saving to file: {e}")

def main():
    input_dir = input("Enter the directory containing the text files: ").strip()
    total_qa_pairs = 5000
    api_key = input("Enter your Groq API key: ").strip()
    output_file_path = os.path.join(os.getcwd(), "qa_pairs.json")

    chunk_method = input("Choose chunking method (characters/words): ").strip().lower()
    if chunk_method == "characters":
        chunk_function = chunk_text_by_characters
    elif chunk_method == "words":
        chunk_function = chunk_text_by_words
    else:
        print("Invalid chunking method selected.")
        return

    if not os.path.isdir(input_dir):
        print(f"The directory at {input_dir} does not exist.")
        return

    text_files = [os.path.join(input_dir, file) for file in os.listdir(input_dir) if file.endswith('.txt')]
    if len(text_files) == 0:
        print("No text files found in the directory.")
        return

    num_pairs_per_file = total_qa_pairs // len(text_files)

    all_qa_pairs = []

    for text_file in text_files:
        try:
            text = extract_text_from_file(text_file)
        except Exception as e:
            print(f"Error reading file {text_file}: {e}")
            continue

        chunks = chunk_function(text)

        for chunk in chunks:
            print(f"Processing chunk of size {len(chunk)} characters from file {text_file}")
            qa_pairs = generate_qa_pairs(chunk, num_pairs_per_file, api_key)
            all_qa_pairs.extend(qa_pairs)

    all_qa_pairs = all_qa_pairs[:total_qa_pairs]  # Ensure we do not exceed the total desired QA pairs

    save_qa_pairs_to_json(all_qa_pairs, output_file_path)

    print(f"QA pairs saved to {output_file_path}")

if __name__ == "__main__":
    main()
