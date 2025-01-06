import os
import re
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import PyPDF2
from transformers import pipeline
import numpy as np

# Initialize the model for vectorization
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize the summarizer
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Initialize Chroma client
client = chromadb.PersistentClient(settings=Settings(anonymized_telemetry=False), path="./chroma_db")

# Name of the collection to delete
collection_name = "documents"

# Check if the collection exists
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

# Function to clean the text by removing unnecessary characters
def clean_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()    # Remove extra spaces
    return text

# Function to load and preprocess documents
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
                    chunks = cleaned_text.split('.')
                    chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
                    docs[file_name] = chunks
                    print(f"Processed document: {file_name} with {len(chunks)} chunks")
            except Exception as e:
                print(f"Error reading PDF {file_name}: {e}")
    return docs

# Function to vectorize documents
def vectorize_documents(preprocessed_documents):
    document_vectors = {}
    for file_name, chunks in preprocessed_documents.items():
        vectors = model.encode(chunks)
        document_vectors[file_name] = vectors
        print(f"Vectorized {len(chunks)} chunks from document: {file_name}")
    return document_vectors

# Function to store vectors in Chroma
def store_vectors_in_chroma(document_vectors):
    for file_name, vectors in document_vectors.items():
        for idx, vector in enumerate(vectors):
            doc_id = f"{file_name}_chunk_{idx+1}"
            collection.add(
                ids=[doc_id],
                documents=[f"Chunk {idx+1} of {file_name}"],
                metadatas=[{"file_name": file_name, "chunk_idx": idx}],
                embeddings=[vector.tolist()]
            )
            print(f"Stored vector for {file_name} chunk {idx+1} in Chroma with ID {doc_id}")

# Function to highlight keywords
def highlight_keywords(text, keyword):
    return text.replace(keyword, f"\033[1;31m{keyword}\033[0m")

# Function to summarize text
def summarize_text(text, max_length=25, min_length=10):
    try:
        if len(text.split()) < min_length:
            return text
        summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        print(f"Error summarizing text: {e}")
        return text

# Function to check similarity threshold
def is_relevant(query, document_text, similarity_threshold=0.5):
    # Convert document text and query into embeddings
    query_vector = model.encode([query])[0]
    document_vector = model.encode([document_text])[0]
    
    # Calculate cosine similarity
    similarity = np.dot(query_vector, document_vector) / (np.linalg.norm(query_vector) * np.linalg.norm(document_vector))
    
    # Return True if similarity exceeds the threshold
    return similarity >= similarity_threshold

# Function to search and filter documents
def search_documents(query, n_results=3, similarity_threshold=0.5):
    query_vector = model.encode([query])

    # Perform the query in Chroma
    results = collection.query(
        query_embeddings=query_vector.tolist(),
        n_results=n_results
    )

    # Filter documents based on relevance to the query
    document_matches = {}
    if results['documents']:
        print(f"Found {len(results['documents'][0])} matching documents for query: '{query}'")

        for i, document in enumerate(results['documents'][0]):
            metadata = results['metadatas'][0][i]
            file_name = metadata.get('file_name', 'Unknown')
            document_text = document

            # Filter documents based on similarity
            if is_relevant(query, document_text, similarity_threshold):
                if file_name not in document_matches:
                    document_matches[file_name] = {'chunks': []}

                document_matches[file_name]['chunks'].append(document)

        # Now print full document content
        if document_matches:
            for file_name, match_info in document_matches.items():
                full_content = " ".join(match_info['chunks'])  # Combine chunks
                highlighted_content = highlight_keywords(full_content, query)
                summary = summarize_text(full_content)

                print(f"\nDocument: {file_name}")
                print(f"Highlighted Content: {highlighted_content}")
                print(f"Summary: {summary}\n")
        else:
            print("No relevant results found.")
    else:
        print("No results found.")


# Main function to preprocess and store in Chroma
def preprocess_and_store_in_chroma(folder_path):
    print("Loading and preprocessing documents...")
    preprocessed_documents = load_documents(folder_path)
    
    print("Vectorizing documents...")
    document_vectors = vectorize_documents(preprocessed_documents)
    
    print("Storing vectors in Chroma...")
    store_vectors_in_chroma(document_vectors)

# Set the folder path containing your documents
folder_path = "data/sample_docs"  # Update this path

# Run the process
preprocess_and_store_in_chroma(folder_path)

# Example query to test the search functionality
search_query = "resume"
search_documents(search_query)