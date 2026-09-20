# Clinical Notes RAG Assistant

A hands-on Retrieval-Augmented Generation (RAG) project that I built to experiment with retrieving relevant clinical-note context before asking an LLM to answer a question.

The project takes structured JSON clinical notes, cleans the useful fields, turns the cleaned notes into embeddings, stores those embeddings in Pinecone, and uses a Streamlit interface to retrieve context and generate a grounded response with Gemini.

> **Important:** This project is for experimentation and demonstration. It is **not a medical diagnostic tool** and should not be used for clinical decisions.

## What I built

The pipeline is split into small steps so each part can be inspected independently:

1. **Clean the source notes** — `clean_files.py`
2. **Create chunks and embeddings** — `ingestion.py`
3. **Test similarity retrieval** — `retrieval.py`
4. **Run the interactive RAG app** — `chatbot_rag.py`

The main idea is simple: instead of sending a question directly to an LLM, the application first searches the vector store for relevant pieces of the clinical notes and then gives those pieces to the model as context.

## Tech stack

- **Python**
- **LangChain**
- **Hugging Face / Sentence Transformers**
- **Pinecone**
- **Google Gemini 2.5 Flash**
- **Streamlit**
- **python-dotenv**
- **MIMIC-IV-Ext Direct clinical notes**

## Project structure

```text
RAG-clinical-notes/
├── dataset/                  # Local clinical-note JSON files
├── Cleaned_Clinical_Notes/   # Generated cleaned text files
├── clean_files.py            # JSON cleaning and preprocessing
├── ingestion.py              # Chunking, embeddings and Pinecone indexing
├── retrieval.py              # Simple retrieval test
├── chatbot_rag.py            # Streamlit RAG application
├── requirements.txt
├── .gitignore
└── README.md
```

## How the pipeline works

### 1. Cleaning

`clean_files.py` walks through the dataset recursively and keeps the clinical context fields used by the application:

- Admission / chief complaint
- Patient history
- Past medical history
- Family history
- Physical examination
- Labs and imaging

Empty values are skipped. Cleaned records are written to `Cleaned_Clinical_Notes/`.

### 2. Chunking and indexing

`ingestion.py` loads the cleaned text files and splits them into overlapping chunks:

- Chunk size: **800 characters**
- Overlap: **400 characters**
- Embedding model: **`sentence-transformers/all-MiniLM-L6-v2`**
- Embedding size: **384**
- Vector database: **Pinecone**
- Similarity metric: **cosine**

Chunk IDs are derived from the source, chunk position and content hash so the same input can be indexed consistently.

### 3. Retrieval

`retrieval.py` is a small command-line check for the vector-search layer. It uses similarity-score filtering and returns the most relevant chunks for a test question.

### 4. Generation

`chatbot_rag.py` provides the Streamlit interface. For each question, it:

1. Searches Pinecone for relevant chunks.
2. Builds a context block from the retrieved notes.
3. Sends the context and question to Gemini.
4. Instructs the model to stay within the supplied clinical context.
5. Shows the retrieved source snippets in the UI.

## Setup

### Requirements

- Python 3.9+
- Pinecone account/API key
- Google Gemini API key

### 1. Clone the repository

```bash
git clone https://github.com/mohdsaif13/RAG-clinical-notes.git
cd RAG-clinical-notes
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

macOS / Linux:

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add environment variables

Create a local `.env` file:

```env
PINECONE_API_KEY=your_pinecone_api_key
GOOGLE_API_KEY=your_gemini_api_key
PINECONE_INDEX_NAME=medical-rag-hf
PINECONE_NAMESPACE=
PINECONE_RESET_NAMESPACE=true
```

Optional Pinecone deployment settings:

```env
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
```

Do **not** commit `.env` or API keys.

## Run it

Run these commands from the repository root.

### Clean the notes

```bash
python clean_files.py
```

### Build the vector index

```bash
python ingestion.py
```

### Test retrieval

```bash
python retrieval.py
```

### Start the application

```bash
streamlit run chatbot_rag.py
```

The Streamlit app normally starts at `http://localhost:8501`.

## Design choices

A few choices in this project are deliberate:

- **Local embedding model:** `all-MiniLM-L6-v2` keeps the embedding step simple and relatively lightweight.
- **Overlapping chunks:** 800/400 helps retain context when a clinical detail sits near a chunk boundary.
- **Similarity thresholding:** the retrieval layer can avoid returning very weak matches.
- **Context-only generation:** the prompt tells Gemini to answer from retrieved notes and say when the information is not available.
- **Environment-based configuration:** API keys and deployment settings stay outside the source code.

## Limitations

This is a portfolio/learning implementation, not a production clinical system.

- Retrieval quality depends on the source notes, chunking strategy and embedding model.
- A similarity match does not mean the retrieved information is clinically correct.
- The LLM can still produce an incorrect or incomplete response.
- There is no clinical validation workflow.
- The application should not be used for diagnosis, treatment or patient-care decisions.

## Data and responsible use

This repository is intended for technical experimentation with clinical-note retrieval. Follow the license, access requirements and usage conditions of the underlying dataset. Avoid adding private patient information, API keys or other sensitive material to the repository.

## What I learned from the project

This project helped me work through the full RAG flow rather than only calling an LLM API:

**data preparation → chunking → embeddings → vector search → context assembly → LLM generation → source display**

That end-to-end workflow is the main reason I keep this project in my portfolio.

## Author

**Md Saif Ali**

Data Science | Machine Learning | Generative AI | RAG

GitHub: https://github.com/mohdsaif13