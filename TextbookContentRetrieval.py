import os
import re
import json
from PyPDF2 import PdfReader
from pyserini.search import SimpleSearcher
from sentence_transformers import SentenceTransformer, util
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Step 1: Define the Tree Node Class
class TreeNode:
    def __init__(self, identifier, content=None):
        self.id = identifier
        self.content = content
        self.children = []
        self.parent = None

    def add_child(self, child_node):
        child_node.parent = self
        self.children.append(child_node)

# Step 2: Create the Hierarchical Tree
def create_hierarchical_tree(text):
    root = TreeNode('root')
    current_node = root

    # Define patterns for chapters and sections (adjust as needed)
    chapter_pattern = re.compile(r'^Chapter \d+')
    section_pattern = re.compile(r'^Section \d+\.\d+')

    for line in text.split('\n'):
        if chapter_pattern.match(line):
            # Create a new chapter node
            chapter_node = TreeNode(line)
            root.add_child(chapter_node)
            current_node = chapter_node
        elif section_pattern.match(line):
            # Create a new section node
            section_node = TreeNode(line)
            current_node.add_child(section_node)
            current_node = section_node
        else:
            # Add content to the current node
            if current_node.content:
                current_node.content += ' ' + line
            else:
                current_node.content = line

    return root

# Step 3: Implement Retrieval Techniques
def bm25_retrieve(query, index_path='index_directory'):
    searcher = SimpleSearcher(index_path)
    hits = searcher.search(query)
    results = [{'docid': hit.docid, 'score': hit.score, 'raw': hit.raw} for hit in hits]
    return results

def bi_encoder_retrieve(query, passages, model_name='sentence-transformers/all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    query_embedding = model.encode(query)
    passage_embeddings = model.encode(passages)
    results = util.semantic_search(query_embedding, passage_embeddings)
    return results

# Collect passages from the hierarchical tree
def collect_passages(node, passages):
    if node.content:
        passages.append(node.content)
    for child in node.children:
        collect_passages(child, passages)

# Step 4: Re-rank the Results
def re_rank_results(bm25_results, bi_encoder_results):
    combined_results = []

    # Create a mapping from docid to content for BM25 results
    bm25_docid_to_content = {result['docid']: result['raw'] for result in bm25_results}

    # Normalize BM25 results
    for result in bm25_results:
        combined_results.append({
            'docid': result['docid'],
            'score': result['score'],
            'content': result['raw'],
            'method': 'bm25'
        })

    # Normalize bi-encoder results
    for result_group in bi_encoder_results:
        for result in result_group:
            docid = result['corpus_id']
            combined_results.append({
                'docid': docid,
                'score': result['score'],
                'content': bm25_docid_to_content.get(str(docid), ''),  # Get content from BM25 results
                'method': 'bi_encoder'
            })

    combined_results.sort(key=lambda x: x['score'], reverse=True)
    return combined_results

# Step 5: Question Answering
model = T5ForConditionalGeneration.from_pretrained('t5-base')
tokenizer = T5Tokenizer.from_pretrained('t5-base')

def generate_answer(query, top_result_context, max_length=512, max_new_tokens=150):
    input_text = f"Question: {query} Context: {top_result_context[:max_length]}"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids

    # Limit the length of input_ids to avoid memory issues
    input_ids = input_ids[:, :max_length]

    output = model.generate(input_ids, max_new_tokens=max_new_tokens)
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    return answer

# PDF text extraction and indexing functions
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text() + '\n'
    return text

def save_extracted_texts(pdf_paths, output_path):
    all_texts = ''
    for pdf_path in pdf_paths:
        content = extract_text_from_pdf(pdf_path)
        all_texts += content
    with open(output_path, 'w', encoding='utf-8') as text_file:
        text_file.write(all_texts)

def index_documents(index_dir, documents):
    if not os.path.exists(index_dir):
        os.makedirs(index_dir)
    
    with open(os.path.join(index_dir, 'docs.jsonl'), 'w', encoding='utf-8') as f:
        for doc_id, doc_text in enumerate(documents):
            doc = {'id': str(doc_id), 'contents': doc_text}
            f.write(json.dumps(doc) + '\n')
    
    os.system(f'python -m pyserini.index.lucene --collection JsonCollection --input {index_dir} --index {index_dir} --generator DefaultLuceneDocumentGenerator --threads 1 --storePositions --storeDocvectors --storeRaw')

# Main script
if __name__ == "__main__":
    pdf_paths = [
        r'C:\Users\MRITUNJAY\Downloads\NLP_Project\Textbooks\Ajay D. Kshemkalyani, Mukesh Singhal - Distributed computing_ principles, algorithms, and systems-Cambridge University Press (2008).pdf',
        r'C:\Users\MRITUNJAY\Downloads\NLP_Project\Textbooks\(Adaptive Computation and Machine Learning) Ralf Herbrich - Learning Kernel Classifiers_ Theory and Algorithms-The MIT Press (2001).pdf',
        r'C:\Users\MRITUNJAY\Downloads\NLP_Project\Textbooks\Joel S. Cohen - Computer algebra and symbolic computation_ elementary algorithms-A K Peters_CRC Press (2002).pdf'
    ]

    output_text_file = 'extracted_texts.txt'
    index_directory = 'index_directory'

    # Step 1: Extract and save texts
    save_extracted_texts(pdf_paths, output_text_file)

    # Step 2: Read the extracted text
    with open(output_text_file, 'r', encoding='utf-8') as file:
        content = file.read()

    # Step 3: Create the hierarchical tree
    tree_root = create_hierarchical_tree(content)

    # Step 4: Collect passages for retrieval
    passages = []
    collect_passages(tree_root, passages)

    # Step 5: Index documents
    index_documents(index_directory, passages)

    # Example query
    query = "distributed computing"
    bm25_results = bm25_retrieve(query, index_directory)
    bi_encoder_results = bi_encoder_retrieve(query, passages)
    combined_results = re_rank_results(bm25_results, bi_encoder_results)

    # Generate an answer based on the top result
    top_result_context = combined_results[0]['content']
    answer = generate_answer(query, top_result_context)
    print(f'Answer: {answer}')
