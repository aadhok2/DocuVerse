import os
import re
import chromadb
from chromadb.config import Settings
import PyPDF2
from sentence_transformers import SentenceTransformer
from transformers import pipeline

class DocumentProcessor:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.client = chromadb.PersistentClient(settings=Settings(anonymized_telemetry=False), path="./chroma_db")
        self.collection_name = "documents"
        self.collection = self._initialize_collection()

    def _initialize_collection(self):
        try:
            collection = self.client.get_collection(self.collection_name)
            self.client.delete_collection(name=self.collection_name)
        except Exception:
            pass
        return self.client.create_collection(self.collection_name)

    def clean_text(self, text):
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        return re.sub(r'\s+', ' ', text).strip()

    def load_documents(self):
        docs = {}
        if not os.path.isdir(self.folder_path):
            raise ValueError(f"The folder path {self.folder_path} does not exist.")
        
        for file_name in os.listdir(self.folder_path):
            file_path = os.path.join(self.folder_path, file_name)
            if os.path.isfile(file_path) and file_name.endswith('.pdf'):
                try:
                    with open(file_path, 'rb') as f:
                        reader = PyPDF2.PdfReader(f)
                        text = ''.join(page.extract_text() for page in reader.pages)
                        cleaned_text = self.clean_text(text)
                        docs[file_name] = cleaned_text
                except Exception as e:
                    print(f"Error reading PDF {file_name}: {e}")
        return docs

    def vectorize_and_store_documents(self, documents):
        for file_name, text in documents.items():
            words = text.split()
            chunks = [' '.join(words[i:i + 1000]) for i in range(0, len(words), 1000)]
            vectors = self.model.encode(chunks)
            for idx, vector in enumerate(vectors):
                doc_id = f"{file_name}_chunk_{idx + 1}"
                self.collection.add(
                    ids=[doc_id],
                    documents=[chunks[idx]],
                    metadatas=[{"file_name": file_name, "chunk_idx": idx}],
                    embeddings=[vector.tolist()]
                )

    def summarize_documents(self, n_results=3):
        query = "Summarize the documents"
        query_vector = self.model.encode([query])
        results = self.collection.query(query_embeddings=query_vector.tolist(), n_results=n_results)

        if not results['documents']:
            return "No relevant documents found for summarization."

        summaries = {}
        for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
            file_name = metadata.get('file_name', 'Unknown')
            document_text = doc.strip()

            if len(document_text.split()) < 100:
                summaries[file_name] = "The document is too short for summarization."
                continue

            try:
                max_length = min(250, len(document_text.split()) // 2)
                min_length = max(50, max_length // 2)
                summary = self.summarizer(document_text, max_length=max_length, min_length=min_length, do_sample=False)
                summaries[file_name] = summaries.get(file_name, "") + " " + summary[0]['summary_text']
            except Exception as e:
                summaries[file_name] = "Error during summarization."

        return [(file_name, summary) for file_name, summary in summaries.items()]

    def run(self):
        print("Loading and processing documents...")
        documents = self.load_documents()
        print("Vectorizing and storing documents...")
        self.vectorize_and_store_documents(documents)
        print("Summarizing documents...")
        summaries = self.summarize_documents(n_results=3)

        if isinstance(summaries, list):
            for file_name, summary in summaries:
                print(f"\nDocument: {file_name}")
                print(f"Summary: {summary}")
        else:
            print(summaries)

if __name__ == "__main__":
    folder_path = "data/sample_docs"
    processor = DocumentProcessor(folder_path)
    processor.run()