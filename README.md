# 🏥 Medical Diagnostic RAG Assistant

This project implements a Retrieval-Augmented Generation (RAG) system to perform diagnostic reasoning over the MIMIC-IV-Ext Direct clinical notes dataset. The system uses a Pinecone Vector Database for efficient context retrieval and the Gemini 2.5 Flash LLM for generating accurate, context-grounded answers to clinical queries.

---

## 🚀 Key Technologies

- **LLM:** Google Gemini 2.5 Flash (via `langchain-google-genai`)
- **Vector Database:** Pinecone
- **Embeddings:** Hugging Face `all-MiniLM-L6-v2` (for speed and cost-efficiency)
- **Framework:** LangChain
- **Frontend:** Streamlit
- **Data:** MIMIC-IV-Ext Direct (JSON format clinical notes)

---

## 📦 Project Structure

```
.
├── Cleaned_Clinical_Notes/
├── dataset/                    # Raw MIMIC-IV JSON files
├── .env                        # Your API keys
├── README.md                   # This file
├── requirements.txt            # Project dependencies
├── clean_files.py              # Script for dataset preprocessing
├── ingestion.py                # Script for RAG pipeline (Indexing)
├── retrieval.py
└── chatbot_rag.py              # Streamlit application (Frontend/Inference)
```

---

## ⚙️ Setup and Installation

### 1. Prerequisites

- Python 3.9+
- A free account on [Pinecone](https://www.pinecone.io/)
- A Google API Key for Gemini (via [Google AI Studio](https://makersuite.google.com/app/apikey))

### 2. Clone Repository and Environment Setup

```bash
# Clone the repository
git clone https://github.com/mohdsaif13/RAG-clinical-notes

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

Install all required libraries specified in `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 4. Configure API Keys

Create a file named `.env` in the root directory and add your keys:

```env
# Replace with your actual keys
PINECONE_API_KEY=YOUR_PINECONE_API_KEY
GOOGLE_API_KEY=YOUR_GEMINI_API_KEY
PINECONE_INDEX_NAME=medical-rag-hf

# Optional. Leave blank to use Pinecone's default namespace.
PINECONE_NAMESPACE=

# Optional. Defaults to true so re-ingestion does not leave old vectors behind.
PINECONE_RESET_NAMESPACE=true
```

---

## 🛠️ Execution Steps (Building the RAG Pipeline)

The system must be built in three sequential stages: **Cleaning**, **Ingestion**, and **Inference**.

### Step 1: Data Cleaning and Preprocessing

The raw MIMIC-IV notes are nested JSON files containing input fields (`input1` through `input6`). Directly indexing these files is inefficient and causes data leakage (since the diagnosis is often in the main JSON key).

The `clean_files.py` script performs the essential preprocessing:

1. **Extraction:** Iterates through all nested JSON files in the `./dataset` directory.
2. **Structuring:** Extracts only the clinical context fields (`input1` through `input6` - History, Vitals, Labs, etc.) and formats them cleanly with headers.
3. **Sanitization:** Ignores empty or "None" values.
4. **Output:** Clears previous generated cleaned files, then saves each patient's combined record into a single `.txt` file in the `./Cleaned_Clinical_Notes` directory. Output filenames include a short hash of the source path, which prevents duplicate case IDs from overwriting each other without exposing diagnosis folder names.

**Run the cleaning script:**

```bash
python clean_files.py
```

### Step 2: Indexing and Embedding (Pinecone Ingestion)

The `ingestion.py` script takes the cleaned text files and converts them into a searchable vector database.

1. **Chunking:** Documents are split into smaller chunks (800 chars with 400 overlap) for better retrieval.
2. **Embedding:** The free and efficient `all-MiniLM-L6-v2` model (384 dimensions) generates numerical vectors.
3. **Uploading:** Existing vectors in the configured namespace are cleared by default, then the current chunks are uploaded to your specified Pinecone index.

**Run the ingestion script:**

```bash
python ingestion.py
```

### Step 3: Run the Streamlit Application (Inference)

The `chatbot_rag.py` script launches the interactive chatbot. It connects to the final Pinecone index, retrieves context based on the user's query, and uses the Gemini LLM to generate the final, grounded response.

**Launch the application:**

```bash
streamlit run chatbot_rag.py
```

The application will automatically open in your web browser (usually at `http://localhost:8501`).

---

## 📝 Notes

- Ensure all API keys are properly configured before running the scripts
- The cleaning step must be completed before ingestion
- Make sure your Pinecone index name matches the one specified in your `.env` file

---

## 🔒 Security Considerations

- Never commit your `.env` file to version control
- Add `.env` to your `.gitignore` file
- Keep your API keys confidential

---





