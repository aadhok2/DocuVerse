import os
import re
import chromadb
from chromadb.config import Settings
import PyPDF2
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import numpy as np

# Step 1: Initialize the model for vectorization and QA pipeline
model = SentenceTransformer('all-MiniLM-L6-v2')  # Sentence transformer for vectorization
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")  # QA pipeline

# Step 2: Initialize Chroma client
client = chromadb.PersistentClient(settings=Settings(anonymized_telemetry=False), path="./chroma_db")
collection_name = "documents"

# Check and delete existing collection if it exists
try:
    collection = client.get_collection(collection_name)
    print(f"The collection '{collection_name}' exists. Deleting it...")
    client.delete_collection(name=collection_name)
    print(f"The collection '{collection_name}' has been deleted.")
except Exception as e:
    print(f"Collection '{collection_name}' does not exist or cannot be accessed. Skipping deletion.")

# Recreate the collection (this clears the previous data)
collection = client.create_collection(collection_name)
print(f"The collection '{collection_name}' has been recreated.")

# Step 3: Function to clean the text (remove special characters, unnecessary spaces)
def clean_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()    # Remove extra spaces
    return text

# Step 4: Function to extract and preprocess PDF documents
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
                    print(f"Processed document: {file_name}")
            except Exception as e:
                print(f"Error reading PDF {file_name}: {e}")
    return docs

# Step 5: Function to vectorize documents and store in Chroma
def vectorize_and_store_documents(documents):
    for file_name, text in documents.items():
        # Split text into chunks (optional, you can split based on content size)
        chunks = text.split('.')
        chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
        
        # Vectorize the chunks
        vectors = model.encode(chunks)
        
        # Store the vectors in Chroma
        for idx, vector in enumerate(vectors):
            doc_id = f"{file_name}_chunk_{idx+1}"
            collection.add(
                ids=[doc_id],
                documents=[f"Chunk {idx+1} of {file_name}"],
                metadatas=[{"file_name": file_name, "chunk_idx": idx}],
                embeddings=[vector.tolist()]
            )
            print(f"Stored vector for {file_name} chunk {idx+1} in Chroma with ID {doc_id}")

# Step 6: Function to answer the query based on stored documents
def answer_query(query, n_results=3):
    # Create a query vector from the question
    query_vector = model.encode([query])
    
    # Perform the query in Chroma to get relevant document chunks
    results = collection.query(
        query_embeddings=query_vector.tolist(),
        n_results=n_results
    )
    print("Retrieved documents:", results['documents'])

    # If no results found, return a message
    if not results['documents']:
        return "No relevant documents found."

    # Prepare to store the answers
    answers = []

    # Iterate over retrieved results and answer based on document context
    for i, document in enumerate(results['documents'][0]):
        metadata = results['metadatas'][0][i]
        file_name = metadata.get('file_name', 'Unknown')
        document_text = document

        # Use the QA model to extract the answer from the document text
        answer = qa_pipeline(question=query, context=document_text)
        if answer['score'] > 0.2:  # Lower confidence threshold for answers
            answers.append((file_name, answer["answer"]))
    
    # Return the top answers
    return answers if answers else ["Could not find a confident answer."]

# Main function to run the entire process
def run_project(folder_path, query):
    print("Loading and processing documents...")
    documents = load_documents(folder_path)
    
    print("Vectorizing and storing documents...")
    vectorize_and_store_documents(documents)
    
    print(f"Answering the query: '{query}'")
    answers = answer_query(query, n_results=3)
    
    if isinstance(answers, list):
        if all(isinstance(a, tuple) and len(a) == 2 for a in answers):  # Checking if it's a list of tuples
            for file_name, answer in answers:
                print(f"\nDocument: {file_name}")
                print(f"Answer: {answer}")
        else:
            print("Unexpected answer format. Here are the raw answers:")
            for answer in answers:
                print(answer)
    else:
        print(answers)

# Path to the folder containing your PDF documents
folder_path = "data/sample_docs"  # Replace with your folder path

# Query to test: asking whose resume is this
query = "What is the address in resume?"

# Run the project
run_project(folder_path, query)