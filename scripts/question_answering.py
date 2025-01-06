import os
import re
import chromadb
from chromadb.config import Settings
import PyPDF2
from sentence_transformers import SentenceTransformer
from transformers import pipeline

model = SentenceTransformer('all-MiniLM-L6-v2')  
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")  

client = chromadb.PersistentClient(settings=Settings(anonymized_telemetry=False), path="./chroma_db")
collection_name = "documents"

try:
    collection = client.get_collection(collection_name)
    client.delete_collection(name=collection_name)
except Exception:
    pass

collection = client.create_collection(collection_name)

def clean_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_documents(folder_path):
    docs = {}
    if not os.path.isdir(folder_path):
        raise ValueError(f"The folder path {folder_path} does not exist.")
    
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path) and file_name.endswith('.pdf'):
            try:
                with open(file_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    text = ''.join(page.extract_text() for page in reader.pages)
                    cleaned_text = clean_text(text)
                    docs[file_name] = cleaned_text
            except Exception as e:
                print(f"Error reading PDF {file_name}: {e}")
    return docs

def vectorize_and_store_documents(documents):
    for file_name, text in documents.items():
        chunks = text.split('.')
        chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
        
        vectors = model.encode(chunks)
        
        for idx, vector in enumerate(vectors):
            doc_id = f"{file_name}_chunk_{idx+1}"
            collection.add(
                ids=[doc_id],
                documents=[f"Chunk {idx+1} of {file_name}"],
                metadatas=[{"file_name": file_name, "chunk_idx": idx}],
                embeddings=[vector.tolist()]
            )

def answer_query(query, n_results=3):
    query_vector = model.encode([query])
    
    results = collection.query(
        query_embeddings=query_vector.tolist(),
        n_results=n_results
    )

    if not results['documents']:
        return "No relevant documents found."

    answers = []

    for i, document in enumerate(results['documents'][0]):
        metadata = results['metadatas'][0][i]
        file_name = metadata.get('file_name', 'Unknown')
        document_text = document

        answer = qa_pipeline(question=query, context=document_text)
        if answer['score'] > 0.2:
            answers.append((file_name, answer["answer"]))
    
    return answers if answers else ["Could not find a confident answer."]

def run_project(folder_path, query):
    print("Loading and processing documents...")
    documents = load_documents(folder_path)
    
    print("Vectorizing and storing documents...")
    vectorize_and_store_documents(documents)
    
    print(f"Answering the query: '{query}'")
    answers = answer_query(query, n_results=3)
    
    if isinstance(answers, list):
        for file_name, answer in answers:
            print(f"\nDocument: {file_name}")
            print(f"Answer: {answer}")
    else:
        print(answers)

folder_path = "data/sample_docs"
query = "who is the person in the story?"

run_project(folder_path, query)