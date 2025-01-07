
# DocuVerse
Document Search and Summarization with Question Answering and Vector Search

This project allows you to load, preprocess, vectorize, and store documents (PDFs) in a Chroma vector database. It leverages Natural Language Processing (NLP) techniques for tasks such as search, document summarization, and question answering using large language models (LLMs). The application uses a combination of pre-trained models for document embeddings, summarization, and QA tasks, and integrates a vector search mechanism to retrieve relevant document chunks based on a query.

## Project Overview

The project aims to:
- Preprocess documents (PDFs) by extracting and cleaning text.
- Vectorize document text into embeddings using the SentenceTransformer model.
- Store these embeddings in a Chroma vector database.
- Use Chroma to search and retrieve relevant document chunks based on query embeddings.
- Implement document summarization and keyword highlighting to improve document relevance.
- Provide question-answering functionality based on document context.

## Key Features
- **Document Preprocessing**: Cleans and extracts text from PDFs, handles chunking of large documents.
- **Vectorization**: Uses `SentenceTransformer` to convert document text into vector embeddings.
- **Chroma Vector Database**: Leverages Chroma to store and query vectorized document data.
- **Search**: Find relevant document chunks based on similarity to a query using cosine similarity.
- **Summarization**: Automatically summarizes document content using `facebook/bart-large-cnn`.
- **Question Answering**: Uses `transformers` pipeline to provide answers from the document context.

## How it Works
1. **Preprocessing**: The application loads PDF files from a specified folder, extracts and cleans text, then splits the text into chunks for vectorization.
2. **Vectorization**: Each chunk is converted into an embedding using a pre-trained `SentenceTransformer` model.
3. **Storage in Chroma**: The document embeddings are stored in a Chroma vector database for efficient querying.
4. **Querying**: Users can input queries, and the system finds the most relevant document chunks based on the query's similarity to stored embeddings.
5. **Summarization**: Once relevant chunks are identified, the system provides a summary for the document.
6. **Question Answering**: Users can ask specific questions, and the system will return an answer based on the document's content.

## Requirements
- Python 3.8+
- Install the required dependencies using the following command:
    ```bash
    pip install -r requirements.txt
    ```

## Setup and Usage
1. Clone the repository:
    ```bash
    git clone <repo-url>
    cd <repo-directory>
    ```

2. Prepare your documents:
    - Place your PDF files in the `data` folder.

3. To perform a search or question answering or summarizer, run:
    ```bash
    python searching_for_file.py
    ```
     ```bash
    python question_answering.py
    ```
    ```bash
    python document_summarizer.py
    ```
   You can change the query inside the `search_or_qa.py` file or modify it to accept dynamic input.

## Matching the Role Requirements

This project demonstrates the following capabilities that match the requirements of the role you are applying for:
1. **Understanding of prompt engineering and prompt tuning**:
   - The project leverages pre-trained models such as `SentenceTransformer` and `facebook/bart-large-cnn` for document vectorization, summarization, and question answering. These models can be fine-tuned or extended based on specific use cases, showcasing an understanding of prompt engineering.

2. **Experience building applications using LLM frameworks such as LangChain, Llama Index, and Semantic Kernel**:
   - While LangChain and Llama Index aren't directly integrated in this version, the core functionalities such as vector search and summarization can be easily extended to include these frameworks. The use of transformers for QA and summarization is also similar in nature to these frameworks.

3. **Experience with vector databases like Faiss or Chroma**:
   - The project uses Chroma as a vector database for storing and querying document embeddings, aligning with the requirement for familiarity with vector databases like Faiss.

4. **Knowledge of ML model evaluation**:
   - The project uses cosine similarity to evaluate the relevance of document chunks to the query, ensuring consistent performance and relevance. Fine-tuning the models or the evaluation strategy can be added to handle different types of documents and queries.

5. **Familiarity with MLOps and ML model lifecycle pipelines**:
   - The project integrates pre-trained models into an end-to-end pipeline from document preprocessing to vectorization, storage, and querying, demonstrating basic MLOps principles. The process can be extended to include model training and fine-tuning.

6. **Experience with ML model training and fine-tuning**:
   - While this project does not include training from scratch, it leverages pre-trained models that can be further fine-tuned or replaced with custom models, matching the requirement for ML model training and fine-tuning experience.

## File Structure
- `data/`: Contains the folder for input PDFs.
- `scripts/`: Contains the core logic for preprocessing, vectorizing, and storing documents in Chroma.
- `app.py`: Main application file for running search queries and question answering.
- `requirements.txt`: List of dependencies required to run the project.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

