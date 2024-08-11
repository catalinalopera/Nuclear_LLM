# %% [markdown]
# ## DOCUMENT

# %%
!pip install faiss-cpu

# %%
import gdown
# Output file name and Google Drive file ID
output = 'Nuclear.zip'
file_id = '1QeYz4v_CNfRF6x8cyowrj7FW9UfLazTm'

# The file is downloaded from Google Drive
gdown.download(id=file_id, output=output, quiet=False)
print('DONE')

import zipfile

zip_file = 'Nuclear.zip'

with zipfile.ZipFile(zip_file, 'r') as zip_ref:
    zip_ref.extractall('./extracted_articles')

print('ZIP was opened succesfully.')

# %%
import os
import zipfile
import urllib.parse

# folder name to create in Colab
extract_folder = '/content/extracted_articles/New folder'

# Create if the folder does not exist
if not os.path.exists(extract_folder):
    os.makedirs(extract_folder)

# Name of the downloaded ZIP file
zip_file = '/content/Nuclear.zip'

# Open ZIP file
with zipfile.ZipFile(zip_file, 'r') as zip_ref:
    zip_ref.extractall(extract_folder)

# Title and article lists
titles = []
articles = []

print('Reading titles...')

i = 0

# Navigating the extracted file folder
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
        articles.append(f.read().decode('utf-8'))  # changed to 'utf-8'
        i += 1

        if i % 500 == 0:
            print('Processed {:,}'.format(i))

print('DONE.\n')
print('There are {:,} articles.'.format(len(articles)))


# %%
titles[0:4]

# %%
articles[0]

# %% [markdown]
# ## CHUNK ARTICLES
# 

# %%
print('Before splitting, {:,} articles.\n'.format(len(titles)))

passage_titles = []
passages = []

print('Splitting...')

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

chunked_corpus = {'title': passage_titles, 'text': passages}

print('Processed {:,} passages.'.format(len(chunked_corpus['title'])))


# %% [markdown]
# ## Create DPR Embeddings

# %%
from transformers import DPRContextEncoder, DPRContextEncoderTokenizerFast
import torch
import time
import datetime
import math
import numpy as np

# Model and tokeniser loading
ctx_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-multiset-base")
ctx_tokenizer = DPRContextEncoderTokenizerFast.from_pretrained("facebook/dpr-ctx_encoder-multiset-base")

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Moving the model to the device
ctx_encoder = ctx_encoder.to(device=device)

# Tokenize 
num_passages = len(chunked_corpus['title'])
print('Tokenizing {:,} passages for DPR...'.format(num_passages))

outputs = ctx_tokenizer(
    chunked_corpus['title'],
    chunked_corpus['text'],
    truncation=True,
    padding='longest',
    max_length=512,
    return_tensors='pt'
)

print("DONE")

input_ids = outputs['input_ids']

# Vectorisation
torch.set_grad_enabled(False)
t0 = time.time()
step = 0
batch_size = 16
num_batches = math.ceil(num_passages / batch_size)
embeds_batches = []
print('Processing {:,} batches...'.format(num_passages))
print(input_ids.shape)

for i in range(0, num_passages, batch_size):
    if step % 100 == 0 and not step == 0:
        elapsed = time.time() - t0
        print('  Batch {:>5} of {:>5} Elapsed {:}'.format(step, num_batches, format_time(elapsed)))

    batch_ids = input_ids[i:i + batch_size].to(device=device)

    outputs = ctx_encoder(
        batch_ids,
        return_dict=True
    )

    embeddings = outputs["pooler_output"]
    embeddings = embeddings.detach().cpu().numpy()
    embeds_batches.append(embeddings)
    step += 1

print('DONE.')

embeddings = np.concatenate(embeds_batches, axis=0)
print('Size of dataset embeddings:', embeddings.shape)


# %% [markdown]
# ## FAISS Index(Facebook Al Similarity Search)
# 

# %%
# FAISS Index
import faiss
import time

import datetime

def format_time(elapsed):
    '''
    Verilen süreyi saniye cinsinden alır ve hh:mm:ss biçiminde döndürür.
    '''
    elapsed_rounded = int(round(elapsed))
    return str(datetime.timedelta(seconds=elapsed_rounded))


# FAISS index settings
dim = 768  # Embedding size
m = 128    # HNSW parameter

# Creating a FAISS index
index = faiss.IndexHNSWFlat(dim, m, faiss.METRIC_INNER_PRODUCT)

# Time measurement
t0 = time.time()

# Training and indexing embeddings
index.train(embeddings)
index.add(embeddings)

# Calculating elapsed time
elapsed = time.time() - t0
print('Done.')
print('Adding embeddings to index took:', format_time(elapsed))


# %%
# Question Encoder Installation and Interrogation
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizerFast

# Question encoder and tokeniser installation
q_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-multiset-base")
q_encoder = q_encoder.to(device=device)

q_tokenizer = DPRQuestionEncoderTokenizerFast.from_pretrained("facebook/dpr-question_encoder-multiset-base")

# Create query embedding
input_ids = q_tokenizer.encode("When the United States tested the first nuclear weapon?", return_tensors="pt")
input_ids = input_ids.to(device)

outputs = q_encoder(input_ids)
q_embed = outputs['pooler_output']
q_embed = q_embed.detach().cpu().numpy()

print("Query embedding:", q_embed.shape)

# Search on the FAISS index
k = 3  # Number of nearest neighbours to rotate
D, I = index.search(q_embed, k=k)

print("Closest matching indices:", I)
print("Inner Products:", D)

# %%
import textwrap

# Wrap text to fit 80 characters
wrapper = textwrap.TextWrapper(width=80)

# For the nearest 'k' result
retrieved_texts = []
for i in I[0]:
    title = chunked_corpus['title'][i]
    passage = chunked_corpus['text'][i]
    retrieved_texts.append(passage)

    print('Index:', i)
    print('Article Title:', title, '\n')
    print('Passage:')
    print(wrapper.fill(passage))
    print(' ')


# %% [markdown]
# ## Ask Questions

# %%
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Model and tokeniser loading
t5_model = T5ForConditionalGeneration.from_pretrained('t5-base')
t5_tokenizer = T5Tokenizer.from_pretrained('t5-base')

# Moving the model to the device
t5_model = t5_model.to(device)


# %%
# Merge undeleted texts
context = " ".join(retrieved_texts)

# Question List
questions = [
    "What groundbreaking discovery did John Cockcroft, Ernest Walton, and Ernest Rutherford make in 1932 regarding lithium atoms?",
    "When the United States tested the first nuclear weapon?",
    "Who discovered the neutron in the same year that induced radioactivity was first observed?",
    "What did Otto Hahn, Fritz Strassmann, Lise Meitner, and Otto Robert Frisch discover in their experiments with neutron-bombarded uranium in 1938?",
    "What did Enrico Fermi focus on in the 1930s to enhance induced radioactivity?",
    "Why did scientists worldwide seek government support for nuclear fission research just before World War II?"
]

# Answer retrieval loop
for question in questions:
    # Combining query and context
    input_text = "question: {} context: {}".format(question, context)

    # Processing input text with tokeniser
    input_ids = t5_tokenizer.encode(input_text, return_tensors='pt').to(device)

    # Creating an answer with the model
    outputs = t5_model.generate(input_ids)
    answer = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("Question:", question)
    print("Generated Answer:", answer)
    print("="*50)  # We add a line to separate questions and answers

# %%
# Merge undeleted texts
context = " ".join(retrieved_texts)

# Question List
questions = [
    "Who were the physicists involved in the first artificial fission of atoms?",
    "What discovery led to the Manhattan Project?",
    "When did John Cockcroft, Ernest Walton, and Ernest Rutherford discover the energy release from splitting lithium atoms?",
    "Who discovered the neutron in 1932?",
    "What did Frédéric and Irène Joliot-Curie discover in 1934?",
    "What did experiments with slow neutrons lead Enrico Fermi to believe he had created?",
    "Who conducted experiments with neutron-bombarded uranium in 1938?",
    "What surprising discovery did Otto Hahn and Fritz Strassmann make in 1938?",
    "Who first recognized the potential of fission reactions for a chain reaction?",
    "When did Frédéric Joliot-Curie announce the experimental confirmation of a self-sustaining nuclear chain reaction?",
    "What was the purpose of the Chicago Pile-1 reactor?",
    "When did Chicago Pile-1 achieve criticality?",
    "What was the goal of the Manhattan Project?",
    "When was the first nuclear weapon tested by the United States?",
    "Who authored the pocketbook 'The Atomic Age' in August 1945?",
    "What did Glenn Seaborg envision for the future of nuclear energy?",
    "What did Eugene Wigner and Alvin Weinberg patent in August 1945?",
    "When was the EBR-I experimental station in Arco, Idaho, first used to generate electricity?",
    "What did Dwight Eisenhower emphasize in his 'Atoms for Peace' speech?",
    "What did the Atomic Energy Act of 1954 enable in the United States?",
    "When did the Obninsk Nuclear Power Plant become the world's first nuclear power plant to generate electricity for a power grid?",
    "Where was the world's first commercial nuclear power station located?",
    "What were the initial capacities of the Calder Hall reactors?",
    "When did the U.S. Army Nuclear Power Program formally commence?",
    "What was the first commercial nuclear station to become operational in the United States?",
    "What impact did the cancellation of a nuclear-powered aircraft carrier contract have on nuclear reactor design adoption?",
    "When was EURATOM launched?",
    "What organization was launched alongside the European Economic Community in 1957?",
    "Where did the first major nuclear reactor accident occur?",
    "What caused the accident at the 3 MW SL-1 reactor?",
    "How many crew fatalities resulted from the accident on the Soviet submarine K-27 in 1968?",
    "How did the global installed nuclear capacity rise from 1960 to the late 1980s?",
    "What was the peak of global nuclear capacity under construction in the late 1970s and early 1980s?",
    "How many nuclear units were cancelled in the United States between 1975 and 1980?",
    "Why was Alvin Weinberg fired from Oak Ridge National Laboratory in 1972?",
    "What was the goal of the test program initiated by Idaho National Laboratory in the late 1970s?",
    "What factors contributed to rising economic costs in nuclear power plant construction during the 1970s and 1980s?",
    "What effect did the 1973 oil crisis have on nuclear power development in France and Japan?",
    "How did the French plan, known as the Messmer plan, aim to reduce dependence on oil?",
    "What role did local opposition play in the cancellation of the proposed Bodega Bay nuclear power station?",
    "When did the first significant anti-nuclear protests emerge in Germany?",
    "What was the significance of the 'Atoms for Peace' speech by Dwight Eisenhower in 1953?",
    "What regulations were implemented in the United States following increased public hostility to nuclear power in the early 1970s?",
    "How did the regulatory changes in the United States affect the license procurement process for nuclear power plants?",
    "How did utility proposals for nuclear generating stations change in the United States between 1974 and 1976?",
    "What was the outcome of the Vermont Yankee Nuclear Power Corp. v. Natural Resources Defense Council, Inc. case?",
    "What was the main consequence of the Three Mile Island accident according to the Nuclear Regulatory Commission?",
    "How did the bankruptcy of Public Service Company of New Hampshire affect the nuclear power industry?",
    "What factors contributed to the shift in electricity generation to coal-fired power plants in the 1980s?",
    "What did President Jimmy Carter call the energy crisis in 1977?",
    "What was the significance of the construction of the first commercial-scale breeder reactor in France?",
    "What was the result of the attack on the Superphenix reactor in France in 1982?",
    "When did the Chernobyl disaster occur?",
    "Where was the Chernobyl Nuclear Power Plant located?",
    "What is considered the worst nuclear disaster in history?",
    "How many personnel were involved in the initial emergency response and decontamination of Chernobyl?",
    "How much did the Chernobyl disaster cost in adjusted inflation terms?",
    "How did the Chernobyl disaster affect the regulation of Western reactors?",
    "What changes were made to the RBMK reactors after the Chernobyl accident?",
    "How many RBMK reactors are still in use today?",
    "What were the primary reasons for the slowdown in global nuclear capacity growth after the late 1980s?",
    "How many nuclear units were cancelled in the United States ultimately?",
    "Who authored the cover story in the 11 February 1985 issue of Forbes magazine criticizing the U.S. nuclear power program?",
    "How did the Three Mile Island accident affect new plant constructions in many countries?",
    "What was the primary cost of implementing regulatory changes after the Three Mile Island accident?",
    "What was the significance of the shutdown of two nuclear power stations in the Tennessee Valley in the 1980s?",
    "What did the anti-nuclear protests in the late 1960s and early 1970s in Europe and North America lead to?",
    "What was the frequency of new nuclear reactor startups globally during the 1980s?",
    "When did utility proposals for nuclear generating stations peak in the United States?",
    "What was the impact of the 1979 oil crisis on countries heavily reliant on oil for electric generation?"
]

# %%
# Answer retrieval loop
for question in questions:
    # Combining query and context
    input_text = "question: {} context: {}".format(question, context)

    # Processing input text with tokeniser
    input_ids = t5_tokenizer.encode(input_text, return_tensors='pt').to(device)

    # Creating an answer with the model
    outputs = t5_model.generate(input_ids)
    answer = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("Question:", question)
    print("Generated Answer:", answer)
    print("="*50)  # We add a line to separate questions and answers

# %%
import time
from tqdm import tqdm  # We add the tqdm library for the progress bar

total_time_start = time.time()  # Start time for total time calculation
total_questions = len(questions)
progress_bar = tqdm(total=total_questions, desc="Progress", position=0)

# Answer retrieval loop
for question in questions:
    # Combining query and context
    input_text = "question: {} context: {}".format(question, context)

    # Processing input text with tokeniser
    input_ids = t5_tokenizer.encode(input_text, return_tensors='pt').to(device)

    # Creating an answer with the model
    start_time = time.time()
    outputs = t5_model.generate(input_ids)
    end_time = time.time()
    answer = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Printing results
    print("Question:", question)
    print("Generated Answer:", answer)
    print("Time taken:", end_time - start_time, "seconds")  # Print answer generation time
    print("=" * 50)  # We add a line to separate questions and answers

    # Update the progress bar
    progress_bar.update(1)

total_time_end = time.time()  # End time for total time calculation
total_execution_time = total_time_end - total_time_start
progress_bar.close()

print("Total Execution Time:", total_execution_time, "seconds")  # Print total time


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%



