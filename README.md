
# Document Summarizer, Searcher, and Question Answering System

This project is a **Document Summarizer**, **File Searcher**, and **Question Answering System** that processes PDF documents, stores them as vectors, and allows searching for relevant documents based on a given query. The documents are stored and indexed using Chroma, and vectors are generated using `SentenceTransformers`. The tool can highlight matching keywords, summarize the content of documents that match the search query, and even answer specific questions based on document content.

## Features

- **Document Summarization**: Extracts and summarizes text from PDF files using the BART model.
- **Search Functionality**: Search documents by providing a query, and get results with the most relevant content.
- **Keyword Highlighting**: Highlight keywords in documents that match the search query.
- **Chroma Database Integration**: Stores document embeddings (vectors) for fast and efficient similarity search.
- **Question Answering**: Provides answers to specific questions based on document content using a question-answering pipeline.

## File Structure

- `README.md`: Project documentation (this file).
- `app.py`: Main application file for searching matching files based on a search query and answering questions.
- `delete_db.py`: Script for deleting the Chroma database.
- `requirements.txt`: List of required Python packages for the project.
- `scripts/`: Directory containing Python scripts for various functionalities (document summarization, vectorization, etc.).
- `data/`: Folder containing input PDF files for processing (documents to be indexed and searched).

## Setup and Installation

### Prerequisites

Ensure you have the following software installed on your system:

- **Python** 3.8 or higher
- **pip** (Python package installer)
- **Git** (optional, for version control)

### Installation Steps

1. **Clone the Repository**:

   If you haven't already, clone the repository to your local machine:

   ```bash
   git clone https://github.com/your-username/document-summarizer-and-searcher.git
   cd document-summarizer-and-searcher
   ```

2. **Install Dependencies**:

   Install the required Python dependencies by running:

   ```bash
   pip install -r requirements.txt
   ```

   The `requirements.txt` file contains the following key dependencies:
   - `sentence-transformers` - For generating document embeddings.
   - `chroma` - For document storage and search.
   - `PyPDF2` - For reading and extracting text from PDF files.
   - `transformers` - For text summarization and question answering using BART.
   - `numpy` - For numerical operations.

3. **Set Up Chroma Database**:

   Chroma is used to store vectorized document embeddings. By default, the Chroma database will be created in a folder named `./chroma_db` in your project directory. You can change the path by modifying the script if needed.

4. **Prepare Documents**:

   Place the PDF documents you want to process inside the `data/` folder. You can add or remove PDF files as required.

   Example structure:

   ```
   data/
     └── sample_docs/
         ├── document1.pdf
         ├── document2.pdf
         └── document3.pdf
   ```

5. **Run the Application**:

   You can now run the main application to preprocess the documents, store the vectors, and search for content:

   ```bash
   python app.py
   ```

   This will:
   - Preprocess the PDF files by extracting and cleaning the text.
   - Vectorize the text using the `SentenceTransformer` model.
   - Store the vectors in the Chroma database.
   - Enable searching for documents using a provided query.

6. **Search Functionality**:

   To perform a search, simply update the `search_query` variable in `app.py` with your desired query. The application will search the indexed documents and return the most relevant results.

   Example:

   ```python
   search_query = "your search query"
   ```

7. **Question Answering**:

   You can also query the documents with a specific question, and the system will attempt to answer based on the document content.

   Example:

   ```python
   question = "What is the main topic of the document?"
   ```

   The application will search for the relevant document(s) and provide an answer based on the content.

8. **Delete Chroma Database** (Optional):

   If you need to clear the Chroma database and start fresh, run the `delete_db.py` script:

   ```bash
   python delete_db.py
   ```

   This will delete the existing Chroma collection and recreate it.

## Usage

1. **Search for Files**:
   - Provide a search query to search through the indexed PDF files.
   - The application will return the most relevant documents based on the query's semantic similarity.

2. **Summarization**:
   - The application automatically summarizes the text from the matched documents.
   - The summary is generated using the BART model.

3. **Keyword Highlighting**:
   - The search results display the document content with matching keywords highlighted.

4. **Question Answering**:
   - The application answers specific questions based on the content of the documents.
   - The question-answering model analyzes the text and provides relevant answers.

## Example Workflow

1. Add PDF files to the `data/sample_docs` folder.
2. Run `app.py` to process and index documents.
3. Change the `search_query` variable in `app.py` to test different search queries.
4. Change the `question` variable in `app.py` to test question answering.
5. Run the application to see results with summarized text, highlighted keywords, and question answering.

## Contributing

Contributions are welcome! Please feel free to fork the repository, create a new branch, and submit a pull request with your changes.

### How to contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature-name`).
3. Make your changes.
4. Commit your changes (`git commit -m 'Add your changes'`).
5. Push to your branch (`git push origin feature/your-feature-name`).
6. Open a pull request with a description of your changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Hugging Face](https://huggingface.co/) for providing the BART model for text summarization and question answering.
- [Sentence-Transformers](https://www.sbert.net/) for the embeddings model.
- [Chroma](https://www.trychroma.com/) for vector database management.
- [PyPDF2](https://github.com/mstamy2/PyPDF2) for PDF text extraction.

---

Feel free to adapt this README according to your specific needs or project updates!
