from sentence_transformers import SentenceTransformer
from chromadb.utils import embedding_to_numpy_array
import chromadb

def create_vector_store(documents):
    # Initialize a ChromaDB client
    client = chromadb.Client()

    # Use a Sentence Transformer model for embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Create a collection in ChromaDB
    collection = client.create_collection("documents")

    for doc_name, doc_content in documents.items():
        # Split content into smaller chunks for better vectorization
        chunks = [doc_content[i:i+500] for i in range(0, len(doc_content), 500)]
        embeddings = model.encode(chunks)
        for i, chunk in enumerate(chunks):
            collection.add(doc_name=f"{doc_name}_chunk{i}", doc_content=chunk, embedding=embedding_to_numpy_array(embeddings[i]))

    return client