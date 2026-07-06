# PDF RAG Chatbot

A simple Retrieval Augmented Generation (RAG) chatbot that allows users to ask questions about a PDF document. The project uses LangChain for orchestration, ChromaDB as the vector database, Hugging Face Sentence Transformers for embeddings, and Gemini 2.5 Flash as the language model.

## Features

* Load and parse PDF documents
* Intelligent document chunking
* Semantic search using vector embeddings
* ChromaDB vector storage with persistence
* Question answering using Gemini 2.5 Flash
* Interactive command line interface
* Displays the source page numbers used to generate each answer

## Tech Stack

* Python
* LangChain
* ChromaDB
* Hugging Face Sentence Transformers
* Google Gemini 2.5 Flash
* PyPDF

## Project Structure

```text
.
├── app.py
├── document.pdf
├── .env
├── chroma_db/
└── README.md
```

## Installation

Clone the repository.

```bash
git clone https://github.com/<your-username>/<repository-name>.git
cd <repository-name>
```

Install the required packages.

```bash
pip install langchain
pip install langchain-community
pip install langchain-core
pip install langchain-text-splitters
pip install langchain-google-genai
pip install langchain-huggingface
pip install langchain-chroma
pip install chromadb
pip install sentence-transformers
pip install pypdf
pip install python-dotenv
```

## Configuration

Create a `.env` file in the project directory.

```env
GOOGLE_API_KEY=YOUR_API_KEY
```

Replace `YOUR_API_KEY` with your Gemini API key.

## Usage

Place your PDF in the project directory and update the `PDF_PATH` variable in `app.py` if necessary.

Run the application.

```bash
python app.py
```

Example session:

```text
===== PDF RAG Chat =====

You: What is Retrieval Augmented Generation?

Answer:
Retrieval Augmented Generation (RAG) combines information retrieval with a large language model to generate responses grounded in external knowledge.

Sources:
Page 3
Page 5
```

Type `exit` or `quit` to end the chat.

## How It Works

1. Load the PDF document.
2. Split the document into overlapping chunks.
3. Generate embeddings for each chunk using a Hugging Face model.
4. Store embeddings in ChromaDB.
5. Embed the user's question.
6. Retrieve the most relevant document chunks.
7. Send the retrieved context and question to Gemini 2.5 Flash.
8. Return the generated answer along with the source pages.

## Future Improvements

* Support multiple PDF documents
* Hybrid search using keyword and semantic retrieval
* Conversation memory
* Source highlighting
* Cross encoder reranking
* Streamlit or Gradio web interface
* Support for additional document formats such as DOCX and TXT

## License

This project is available under the MIT License.
