import streamlit as st
import os
from dotenv import load_dotenv

# import pinecone
from pinecone import Pinecone

# import langchain
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

load_dotenv()

st.set_page_config(page_title="Medical Diagnostic Assistant")
st.title("🏥 Medical Diagnostic Assistant")
st.caption("RAG Chatbot using Pinecone & MIMIC-IV Clinical Notes")

pinecone_api_key = os.environ.get("PINECONE_API_KEY")
google_api_key = os.environ.get("GOOGLE_API_KEY")
index_name = os.environ.get("PINECONE_INDEX_NAME")
pinecone_namespace = os.environ.get("PINECONE_NAMESPACE", "").strip()

missing_settings = [
    name
    for name, value in {
        "PINECONE_API_KEY": pinecone_api_key,
        "GOOGLE_API_KEY": google_api_key,
        "PINECONE_INDEX_NAME": index_name,
    }.items()
    if not value
]

if missing_settings:
    st.error(f"Missing required environment setting(s): {', '.join(missing_settings)}")
    st.stop()

try:
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(index_name)
except Exception as e:
    st.error(f"Failed to initialize Pinecone index: {e}")
    st.stop()

# initialize embeddings model + vector store
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

embeddings = get_embeddings()

vector_store = PineconeVectorStore(
    index=index,
    embedding=embeddings,
    namespace=pinecone_namespace or None,
)

@st.cache_resource
def load_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", # Fast and efficient
        google_api_key=google_api_key,
        temperature=0.3,          # Low temperature for factual medical answers
        max_tokens=512,
        timeout=None,
        max_retries=2,
    )
try:
    llm = load_llm()
except Exception as e:
    st.error(f"Failed to initialize Gemini LLM: {e}")
    st.stop()
# initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Add a system message to set the behavior
    intro_prompt = SystemMessage(
        content="You are a specialized Medical Assistant. You answer questions based strictly on the provided clinical notes. "
                "If the information is not in the context, say you don't know."
    )
    st.session_state.messages.append(intro_prompt)

# display chat messages from history on app rerun
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

# Chat Input Bar
if query := st.chat_input("E.g., What were the patient's vitals upon admission?"):

    # 1. Display and save User Message
    with st.chat_message("user"):
        st.markdown(query)
    st.session_state.messages.append(HumanMessage(content=query))

    # 2. Process Response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing patient records..."):
            
            # A. Retrieve relevant documents from Pinecone
            retriever = vector_store.as_retriever(
                # Search for 4 relevant chunks with a minimum score of 0.3
                search_type="similarity_score_threshold",
                search_kwargs={"k": 4, "score_threshold": 0.3},
            )
            try:
                docs = retriever.invoke(query)
            except Exception as e:
                st.error(f"Retrieval Error: {e}")
                answer_text = "I could not retrieve relevant clinical notes because the vector database query failed."
                st.markdown(answer_text)
                st.session_state.messages.append(AIMessage(content=answer_text))
                st.stop()
            
            # Combine retrieved text into a single context block
            docs_text = "\n\n".join([f"--- Source {i+1} ({d.metadata.get('source', 'Unknown')}): {d.page_content}" for i, d in enumerate(docs)])
            
            if not docs:
                answer_text = "I do not know because no relevant clinical notes were found for this query."
                st.markdown(answer_text)
                st.warning("No relevant clinical notes found for this query.")
                st.session_state.messages.append(AIMessage(content=answer_text))
            else:
                # B. Construct the Augmented Prompt
                rag_prompt_content = f"""You are a helpful Medical Diagnostic Assistant.

Task: Answer the user's question using ONLY the following context provided under 'Context from Clinical Notes:'.
If the answer is not present in the context, strictly state that you do not know.
Use three sentences maximum and keep the answer concise, focusing on clarity.

Context from Clinical Notes:
{docs_text}

User Question:
{query}
"""

                # C. Send to Gemini with only the system instruction and current retrieved context.
                messages_for_llm = [st.session_state.messages[0], HumanMessage(content=rag_prompt_content)]

                try:
                    response = llm.invoke(messages_for_llm)
                    answer_text = response.content

                    # D. Display Answer
                    st.markdown(answer_text)

                    # E. Display Sources
                    with st.expander("📚 View Source Documents"):
                        for i, doc in enumerate(docs):
                            source_name = doc.metadata.get('source', 'Unknown')
                            st.markdown(f"**Source {i+1}:** `{source_name}`")
                            st.caption(doc.page_content[:400] + "...")
                            st.divider()

                    # F. Save Assistant Response to History
                    st.session_state.messages.append(AIMessage(content=answer_text))

                except Exception as e:
                    st.error(f"Generation Error: {e}")
