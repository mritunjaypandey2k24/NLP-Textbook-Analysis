# StepsAI-NLP-Project

# Textbook Content Retrieval and Question Answering System

This project involves developing a system for extracting content from textbooks, organizing it hierarchically, indexing the text, and implementing a retrieval system to answer questions based on the extracted content. The project leverages various NLP techniques, including BM25 retrieval, bi-encoder retrieval, and a T5-based question-answering system.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Methods](#methods)
- [Evaluation](#evaluation)
- [Contributors](#contributors)
- [License](#license)

## Introduction

The goal of this project is to develop a system that can extract and organize content from textbooks, index the content for efficient retrieval, and provide answers to user queries based on the extracted content. The system uses various natural language processing techniques to achieve these objectives.

## Features

- Extract text from PDF textbooks
- Organize content hierarchically into chapters and sections
- Index text using BM25 for efficient retrieval
- Implement bi-encoder retrieval for enhanced search
- Re-rank results from different retrieval methods
- Answer user queries using a T5-based question-answering model

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/NLP_Project.git
    cd NLP_Project
    ```

2. Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Download the necessary models:
    ```python
    from transformers import T5ForConditionalGeneration, T5Tokenizer
    T5ForConditionalGeneration.from_pretrained('t5-base')
    T5Tokenizer.from_pretrained('t5-base')

    from sentence_transformers import SentenceTransformer
    SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    ```

## Usage

1. Extract text from PDFs and save it:
    ```python
    python extract_texts.py
    ```

2. Create the hierarchical tree structure:
    ```python
    python create_hierarchical_tree.py
    ```

3. Index the documents:
    ```python
    python index_documents.py
    ```

4. Perform a query and get an answer:
    ```python
    python query_system.py "Your question here"
    ```

### Example Query

To retrieve information on distributed computing:
```python
query = "distributed computing"
bm25_results = bm25_retrieve(query, 'index_directory')
bi_encoder_results = bi_encoder_retrieve(query, passages)
combined_results = re_rank_results(bm25_results, bi_encoder_results)
top_result_context = combined_results[0]['content']
answer = generate_answer(query, top_result_context)
print(f'Answer: {answer}')
```

## Project Structure

- `extract_texts.py`: Script for extracting text from PDF files.
- `create_hierarchical_tree.py`: Script for creating a hierarchical tree of the text.
- `index_documents.py`: Script for indexing the extracted texts.
- `query_system.py`: Script for querying the system and generating answers.
- `index_directory/`: Directory where indexed files are stored.
- `extracted_texts.txt`: File containing extracted texts from PDFs.

## Methods

### Text Extraction

Extract text from PDF textbooks using PyPDF2.

### Hierarchical Indexing

Organize the extracted text into a hierarchical tree with chapters and sections.

### Retrieval

- **BM25 Retrieval**: Use BM25 algorithm for initial text retrieval.
- **Bi-encoder Retrieval**: Use SentenceTransformer for semantic search.

### Re-ranking

Combine and re-rank the results from BM25 and bi-encoder retrieval methods.

### Question Answering

Use a T5-based model to generate answers based on the top retrieved results.

## Evaluation

The system can be evaluated by comparing the accuracy and relevance of the answers generated for a set of predefined questions based on the textbooks used.

## Contributors

- Mritunjay Pandey

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```
