# %%
!pip install -q faiss-cpu transformers torch gdown

# %%
import os
import zipfile
import urllib.parse
import gdown
import time
import datetime
import textwrap
import math
import numpy as np
import faiss
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, DPRContextEncoder, DPRContextEncoderTokenizerFast, DPRQuestionEncoder, DPRQuestionEncoderTokenizerFast
from tqdm import tqdm
import re

# %%
# Output file name and Google Drive file ID
output = 'Nuclear.zip'
file_id = '1QeYz4v_CNfRF6x8cyowrj7FW9UfLazTm'

# The file is downloaded from Google Drive
gdown.download(id=file_id, output=output, quiet=False)
print('DONE')

zip_file = 'Nuclear.zip'

with zipfile.ZipFile(zip_file, 'r') as zip_ref:
    zip_ref.extractall('./extracted_articles')

print('ZIP was opened successfully.')

# Folder name to be created
extract_folder = './extracted_articles/New_folder'

# Create the folder if it does not exist
if not os.path.exists(extract_folder):
    os.makedirs(extract_folder)

# Extract the ZIP file
with zipfile.ZipFile(zip_file, 'r') as zip_ref:
    zip_ref.extractall(extract_folder)


# %%
# Folder name to be created in Colab
extract_folder = '/content/extracted_articles/New folder'

# Create the folder if it does not exist
if not os.path.exists(extract_folder):
    os.makedirs(extract_folder)

# Extract the ZIP file
with zipfile.ZipFile(zip_file, 'r') as zip_ref:
    zip_ref.extractall(extract_folder)

# Lists for titles and articles
titles = []
articles = []

print('Reading titles...')

i = 0

# Traverse the extracted files in the folder
for filename in os.listdir(extract_folder):
    if not filename.endswith('.txt'):
        continue

    file_path = os.path.join(extract_folder, filename)

    with open(file_path, 'rb') as f:
        title = urllib.parse.unquote(filename[:-4])
        title = title.replace('_', ' ')

        if len(title) == 0 or len(title.strip()) == 0:
            print('Empty title for', filename)
            continue

        titles.append(title)
        articles.append(f.read().decode('utf-8'))  # Changed to 'utf-8'
        i += 1

        if i % 500 == 0:
            print('Processed {:,}'.format(i))

print('DONE.\n')
print('There are {:,} articles.'.format(len(articles)))

# %%
# Before splitting
print('Before splitting, {:,} articles.\n'.format(len(titles)))

passage_titles = []
passages = []

print('Splitting...')

# Splitting articles into chunks
for i in range(len(titles)):
    title = titles[i]
    article = articles[i]

    if len(article) == 0:
        print('Empty article for', title)
        continue

    words = article.split()

    for j in range(0, len(words), 100):
        chunk_words = words[j:j+100]
        chunk = " ".join(chunk_words).strip()

        if len(chunk) == 0:
            continue

        passage_titles.append(title)
        passages.append(chunk)

print('DONE.\n')

# Creating chunked_corpus dictionary
chunked_corpus = {'title': passage_titles, 'text': passages}

print('Processed {:,} passages.'.format(len(chunked_corpus['title'])))


# %%
import torch
from transformers import DPRContextEncoder, DPRContextEncoderTokenizerFast

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the DPR context encoder and tokenizer
ctx_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-multiset-base")
ctx_encoder = ctx_encoder.to(device)

ctx_tokenizer = DPRContextEncoderTokenizerFast.from_pretrained("facebook/dpr-ctx_encoder-multiset-base")

# Encode the paragraphs
encoded_input = ctx_tokenizer(chunked_corpus['text'], padding=True, truncation=True, return_tensors='pt')
encoded_input = encoded_input.to(device)

# Compute the embeddings
with torch.no_grad():
    embeddings = ctx_encoder(encoded_input['input_ids']).pooler_output

embeddings = embeddings.detach().cpu().numpy()

print('Computed embeddings for the paragraphs.')



# %%
import faiss
import numpy as np

# Normalize embeddings
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

# FAISS index creation
dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)  # Using Inner Product (dot product) for similarity
index.add(embeddings)

print('Created FAISS index and added embeddings.')

# %%
# Nearest neighbor search for a sample query
query_embedding = np.random.rand(1, dim).astype('float32')
query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
D, I = index.search(query_embedding, k=10)

print('Nearest neighbors:', I)
print('Distances:', D)

# Print out the passages and their sources
for idx in I[0]:
    print(f"Paragraph Source: {chunked_corpus['title'][idx]}")
    print(f"Paragraph: {chunked_corpus['text'][idx]}\n")

# %%
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizerFast

# Load question encoder and tokenizer
q_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-multiset-base")
q_encoder = q_encoder.to(device)

q_tokenizer = DPRQuestionEncoderTokenizerFast.from_pretrained("facebook/dpr-question_encoder-multiset-base")

# Create query embeddings
questions = [
    "When did the United States test the first nuclear weapon?",
    "What are the causes of climate change?",
    "Explain the process of photosynthesis."
]

input_ids = q_tokenizer(questions, return_tensors="pt", padding=True, truncation=True)
input_ids = input_ids.to(device)

outputs = q_encoder(input_ids['input_ids'])
q_embeds = outputs['pooler_output']
q_embeds = q_embeds.detach().cpu().numpy()

# Normalize embeddings (optional)
q_embeds = q_embeds / np.linalg.norm(q_embeds, axis=1, keepdims=True)

# Search on the FAISS index
k = 3  # Number of nearest neighbors to return
D, I = index.search(q_embeds, k=k)

# Wrap texts to fit within 80 characters
import textwrap
wrapper = textwrap.TextWrapper(width=80)

# Print out the results
print("\n======================== Question and Answer Retrieval ========================\n")

for idx, question in enumerate(questions):
    print(f"Question: {question}\n")
    for rank, i in enumerate(I[idx]):
        passage = chunked_corpus['text'][i]  # Access text from chunked_corpus using index i
        print(f"Rank {rank + 1}")
        print('Index:', i)
        print('Source:', chunked_corpus['title'][i])  # Print the source of the paragraph
        print('Passage:')
        print(wrapper.fill(passage))
        print("\n")
    print("=" * 80 + "\n")

print("Closest matching indices:", I)
print("Inner Products:", D)


# %%
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Initialize T5 model and tokenizer
model_name = "t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# %%
# Function to find closest passage in Faiss index
def find_closest_passage(query_embedding):
    query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
    D, I = index.search(query_embedding, k=1)
    return I[0][0], chunked_corpus['text'][I[0][0]]

# Function to process question and generate answer
def process_question(question):
    # Encode the question using DPR Context Encoder
    encoded_question = ctx_tokenizer(question, return_tensors="pt", padding=True, truncation=True)
    encoded_question = encoded_question.to(device)

    with torch.no_grad():
        question_embedding = ctx_encoder(encoded_question['input_ids']).pooler_output
        question_embedding = question_embedding.detach().cpu().numpy()

    # Find closest passage using Faiss index
    closest_idx, closest_passage = find_closest_passage(question_embedding)

    # Generate answer using T5 model
    input_text = "question: {} context: {}".format(question, closest_passage)
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True).to(device)

    with torch.no_grad():
        outputs = model.generate(input_ids)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return question, answer, closest_passage

# Example usage
question = "What is nuclear energy?"
question_text, answer, context = process_question(question)

# Print results
print("Question:", question_text)
print("Answer:", answer)
print("Context:\n", textwrap.fill(context, width=80))

# %%
question = "What is nuclear energy?"
answer = "providing clean power while also reversing the impact fossil fuels have had on our climate"
context = """The same time, some Asian countries, such as China and India, have committed to rapid expansion of nuclear power. In other countries, such as the United Kingdom and the United States, nuclear power is planned to be part of the energy mix together with renewable energy. Nuclear energy may be one solution to providing clean power while also reversing the impact fossil fuels have had on our climate. These plants would capture carbon dioxide and create a clean energy source with zero emissions, making a carbon-negative process. Scientists propose that 1.8 million lives have already been saved by replacing fossil fuels."""

def prepare_input(question, answer, context):
    input_text = f"question: {question} answer: {answer} context: {context}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True).to(device)
    return input_ids

input_ids = prepare_input(question, answer, context)

def generate_complex_answer(input_ids):
    with torch.no_grad():
        outputs = model.generate(input_ids, max_length=300, num_beams=5, early_stopping=True)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

complex_answer = generate_complex_answer(input_ids)
print("Complex Answer:", complex_answer)


