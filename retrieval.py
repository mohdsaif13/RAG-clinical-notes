import os

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone


load_dotenv()

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def required_env(name):
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"Missing required environment setting: {name}")
    return value


pinecone_api_key = required_env("PINECONE_API_KEY")
index_name = required_env("PINECONE_INDEX_NAME")
pinecone_namespace = os.environ.get("PINECONE_NAMESPACE", "").strip()

pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index(index_name)

print(f"Loading embeddings: {EMBEDDING_MODEL}")
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},
)
vector_store = PineconeVectorStore(
    index=index,
    embedding=embeddings,
    namespace=pinecone_namespace or None,
)

retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 5, "score_threshold": 0.6},
)

query = "What are the symptoms of aortic dissection?"
print(f"Query: {query}")

try:
    results = retriever.invoke(query)

    print(f"\nFound {len(results)} relevant document(s):\n")
    for i, result in enumerate(results):
        source = result.metadata.get("source", "Unknown Source")
        print(f"--- Result {i + 1} (Source: {source}) ---")
        print(result.page_content[:200] + "...\n")

except Exception as e:
    print(f"Error during retrieval: {e}")
