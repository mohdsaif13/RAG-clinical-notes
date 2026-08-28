import hashlib
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec


load_dotenv()

DATA_PATH = Path("./Cleaned_Clinical_Notes")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384


def required_env(name):
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"Missing required environment setting: {name}")
    return value


def setting_enabled(name, default=True):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def index_names(indexes):
    names = []
    for index_info in indexes:
        names.append(index_info["name"] if isinstance(index_info, dict) else index_info.name)
    return names


def index_ready(pinecone_client, name):
    status = pinecone_client.describe_index(name).status
    if isinstance(status, dict):
        return status.get("ready", False)
    return status.ready


def load_cleaned_documents():
    if not DATA_PATH.exists():
        raise SystemExit("Cleaned notes folder not found. Run `python clean_files.py` first.")

    loader = DirectoryLoader(
        str(DATA_PATH),
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    documents = loader.load()

    if not documents:
        raise SystemExit("No cleaned note files found. Run `python clean_files.py` first.")

    return documents


def build_chunk_ids(documents):
    chunk_ids = []

    for chunk_number, document in enumerate(documents):
        source = document.metadata.get("source", "unknown-source")
        start_index = document.metadata.get("start_index", chunk_number)
        id_text = f"{source}:{start_index}:{document.page_content}"
        digest = hashlib.sha1(id_text.encode("utf-8")).hexdigest()
        chunk_ids.append(f"clinical-note-{digest}")

    return chunk_ids


pinecone_api_key = required_env("PINECONE_API_KEY")
index_name = required_env("PINECONE_INDEX_NAME")
pinecone_namespace = os.environ.get("PINECONE_NAMESPACE", "").strip()
pinecone_cloud = os.environ.get("PINECONE_CLOUD", "aws")
pinecone_region = os.environ.get("PINECONE_REGION", "us-east-1")
reset_namespace = setting_enabled("PINECONE_RESET_NAMESPACE", default=True)

pc = Pinecone(api_key=pinecone_api_key)

if index_name not in index_names(pc.list_indexes()):
    pc.create_index(
        name=index_name,
        dimension=EMBEDDING_DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(cloud=pinecone_cloud, region=pinecone_region),
    )
    while not index_ready(pc, index_name):
        time.sleep(1)

index = pc.Index(index_name)

print(f"Loading documents from {DATA_PATH}")
raw_documents = load_cleaned_documents()
print(f"Loaded {len(raw_documents)} file(s).")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=400,
    length_function=len,
    is_separator_regex=False,
    add_start_index=True,
)
documents = text_splitter.split_documents(raw_documents)
chunk_ids = build_chunk_ids(documents)

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},
)

if reset_namespace:
    namespace_label = pinecone_namespace or "default"
    print(f"Clearing existing vectors from the {namespace_label} namespace.")
    delete_args = {"delete_all": True}
    if pinecone_namespace:
        delete_args["namespace"] = pinecone_namespace
    index.delete(**delete_args)

vector_store = PineconeVectorStore(
    index=index,
    embedding=embeddings,
    namespace=pinecone_namespace or None,
)

print(f"Uploading {len(documents)} chunk(s) to Pinecone index '{index_name}'.")
vector_store.add_documents(documents=documents, ids=chunk_ids)
print("Ingestion complete.")
