This repository contains a script that automates the extraction of text from PDF files, splits the text into manageable chunks, generates question-answer (QA) pairs using the OpenAI language model, and saves the results to a JSON file. The script uses a paid OpenAI API key.

## Features

- Extracts text from PDF files in a specified directory.
- Splits extracted text into smaller, manageable chunks.
- Generates QA pairs using the OpenAI language model.
- Limits the number of QA pairs per chunk.
- Saves generated QA pairs in a JSON format suitable for fine-tuning large language models (LLMs).
- Includes basic error handling to manage issues with PDF file processing.

## Requirements

- Python 3.6 or higher
- OpenAI API key (paid)
- Required Python packages:
  - openai
  - langchain
  - pandas
  - tqdm
  - PyMuPDF
  - openpyxl
