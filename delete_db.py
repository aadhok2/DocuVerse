# import chromadb
# from chromadb.config import Settings

# # Initialize the Chroma client
# client = chromadb.PersistentClient(settings=Settings(anonymized_telemetry=False), path="./chroma_db")

# # Name of the collection to delete
# collection_name = "documents_v2"

# # Delete the collection
# client.delete_collection(name=collection_name)

# print(f"Collection '{collection_name}' has been deleted.")
import nltk
nltk.download('punkt')