import os
import re
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import PyPDF2

# Initialize the model for vectorization
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize Chroma client
client = chromadb.PersistentClient(settings=Settings(anonymized_telemetry=False), path="./chroma_db")

# Name of the collection to delete
collection_name = "documents"

# Check if the collection exists
try:
    # Attempt to get the collection
    collection = client.get_collection(collection_name)
    print(f"The collection '{collection_name}' exists. Deleting it...")

    # Delete the collection
    client.delete_collection(name=collection_name)
    print(f"The collection '{collection_name}' has been deleted.")
except Exception as e:
    print(f"Collection '{collection_name}' does not exist or cannot be accessed. Skipping deletion.")

# Recreate the collection (this clears the previous data)
collection = client.create_collection(collection_name)
print(f"The collection '{collection_name}' has been recreated.")

# Function to clean the text by removing unnecessary characters
def clean_text(text):
    # Remove special characters and extra spaces
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Function to load and preprocess documents

def load_documents(folder_path):
    docs = {}
    if not os.path.isdir(folder_path):
        raise ValueError(f"The folder path {folder_path} does not exist.")
    
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            if file_name.endswith('.pdf'):
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
            else:
                print(f"Skipping non-PDF file: {file_name}")
    return docs

# Function to vectorize documents using SentenceTransformer
def vectorize_documents(preprocessed_documents):
    """
    Convert the preprocessed document chunks into vectors using a transformer model.
    """
    document_vectors = {}

    for file_name, chunks in preprocessed_documents.items():
        # Encode each chunk into a vector using the sentence transformer model
        vectors = model.encode(chunks)
        document_vectors[file_name] = vectors
        print(f"Vectorized {len(chunks)} chunks from document: {file_name}")

    return document_vectors

# Function to store vectors in Chroma
def store_vectors_in_chroma(document_vectors):
    for file_name, vectors in document_vectors.items():
        for idx, vector in enumerate(vectors):
            # Create a unique ID for each document chunk (you can use a combination of file_name and chunk_idx)
            doc_id = f"{file_name}_chunk_{idx+1}"
            
            # Add to the Chroma collection with a unique ID
            collection.add(
                ids=[doc_id],  # Unique ID for each chunk
                documents=[f"Chunk {idx+1} of {file_name}"],
                metadatas=[{"file_name": file_name, "chunk_idx": idx}],
                embeddings=[vector.tolist()]  # Chroma expects vectors to be in list form
            )
            print(f"Stored vector for {file_name} chunk {idx+1} in Chroma with ID {doc_id}")

# Function to search for documents based on a query
def search_documents(query, n_results=3):
    query_vector = model.encode([query])  # Encode the query into a vector

    # Perform the query
    results = collection.query(
        query_embeddings=query_vector.tolist(),
        n_results=n_results
    )

    # Debugging: Print raw results
    print("Raw results:", results)

    # Check if the results contain documents
    if results['documents'] and len(results['documents'][0]) > 0:
        print(f"Found {len(results['documents'][0])} matching documents for query: '{query}'")

        # Iterate over results
        for i, document in enumerate(results['documents'][0]):
            try:
                # Access corresponding metadata
                metadata = results['metadatas'][0][i] if results['metadatas'] and len(results['metadatas'][0]) > i else {}
                file_name = metadata.get('file_name', 'Unknown')
                chunk_idx = metadata.get('chunk_idx', 'Unknown')

                # Print the results
                print(f"Document {i + 1}: {file_name} (chunk {chunk_idx})")
                print(f"Content: {document}")
            except Exception as e:
                print(f"Error processing result {i + 1}: {e}")
    else:
        print("No results found.")

# Main function to run preprocessing, vectorization, and storing in Chroma
def preprocess_and_store_in_chroma(folder_path):
    print("Loading and preprocessing documents...")
    preprocessed_documents = load_documents(folder_path)
    
    print("Vectorizing documents...")
    document_vectors = vectorize_documents(preprocessed_documents)
    
    print("Storing vectors in Chroma...")
    store_vectors_in_chroma(document_vectors)

# Set the folder path containing your documents
folder_path = "../data/sample_docs"  # Update this path

# Run the process
preprocess_and_store_in_chroma(folder_path)

# Example query to test the search functionality
search_query = "docusign"
search_documents(search_query)