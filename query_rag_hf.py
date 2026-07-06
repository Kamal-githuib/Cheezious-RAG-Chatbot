import streamlit as st
import os
import re
import fitz
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import google.generativeai as genai
import uuid
from dotenv import load_dotenv

# ==============================
# LOAD ENV
# ==============================
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    st.error("⚠ GEMINI_API_KEY environment variable missing")
else:
    genai.configure(api_key=GEMINI_API_KEY)

# ==============================
# CONFIG
# ==============================
st.set_page_config(page_title="Cheezious", layout="centered")

PDF_DIR = "pdfs"
CHROMA_DB_DIR = "./chroma_db"
COLLECTION_NAME = "psy_docs"

embedder = SentenceTransformer("all-mpnet-base-v2")
client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
collection = client.get_or_create_collection(COLLECTION_NAME)

# ==============================
# PDF FUNCTIONS
# ==============================
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def chunk_text(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def ingest_pdfs():
    if collection.count() > 0:
        st.info("Documents already ingested.")
        return
    pdf_files = [f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")]
    if not pdf_files:
        st.warning("No PDFs found in 'pdfs/' folder.")
        return

    # Show a spinner while loading PDFs
    with st.spinner("Loading PDFs and creating embeddings, please wait..."):
        for pdf_file in pdf_files:
            path = os.path.join(PDF_DIR, pdf_file)
            text = extract_text_from_pdf(path)
            chunks = chunk_text(text)
            for i, chunk in enumerate(chunks):
                emb = embedder.encode(chunk).tolist()
                collection.add(
                    ids=[str(uuid.uuid4())],
                    documents=[chunk],
                    embeddings=[emb],
                )
    st.success("✅ PDFs ingested into ChromaDB!")

# ==============================
# RAG FUNCTIONS
# ==============================
def retrieve(query, top_k=15):
    q_emb = embedder.encode(query).tolist()
    results = collection.query(
        query_embeddings=[q_emb],
        n_results=top_k,
        include=['documents','metadatas']
    )
    docs = []
    for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
        docs.append({'text': doc, 'meta': meta})
    return docs

def build_prompt(retrieved_docs, user_query):
    ctx = ""
    for d in retrieved_docs:
        meta = d.get("meta") or {}  # ✅ safe guard if metadata missing
        ctx += f"[Doc: {meta.get('doc_id','unknown')} | Section: {meta.get('section','')}] \n{d['text']}\n\n"

    system = (
        "You are an assistant for fast food queries. You MUST only use the information from the CONTEXT below. "
        "If the answer is not contained in CONTEXT, say: 'I cannot find the answer to this question in our current documents. "
        "Please check the menu or contact the restaurant.' "
        "Always include which document/section you used (doc_id) at the end. "
        "Do not add or invent prices, deals, or menu items. "
        "Provide accurate information about pizza sizes, burger rates, combo deals, and any other fast food items. "
        "Do not reply to any harmful or inappropriate queries."
    )
    prompt = f"SYSTEM:\n{system}\n\nCONTEXT:\n{ctx}\nUSER: {user_query}\n"
    return prompt

def detect_self_harm(query):
    danger_keywords = ["suicide", "kill myself", "end my life", "khudkushi", "self harm"]
    return any(re.search(rf"\b{kw}\b", query.lower()) for kw in danger_keywords)

def llm_answer(prompt):
    response = genai.GenerativeModel("gemini-2.5-flash").generate_content(prompt)
    return response.text.strip()

# ==============================
# STREAMLIT UI
# ==============================
st.title("Cheezious")
st.write("This chatbot only provides answers from approved Cheezious documents.")

# Ingest PDFs automatically on app start
if "pdfs_loaded" not in st.session_state:
    ingest_pdfs()
    st.session_state.pdfs_loaded = True

# Initialize chat messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if user_query := st.chat_input("Type your question here..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    if detect_self_harm(user_query):
        bot_reply = (
            "⚠ If you are having thoughts of harming yourself, "
            "Please immediately contact your clinician or emergency services. "
            "It is very important that you seek help right away. 🚨"
        )
    else:
        # Show spinner while generating answer
        with st.spinner("Generating answer, please wait..."):
            retrieved = retrieve(user_query, top_k=15)
            prompt = build_prompt(retrieved, user_query)
            bot_reply = llm_answer(prompt)

    st.session_state.messages.append({"role": "assistant", "content": bot_reply})
    with st.chat_message("assistant"):
        st.markdown(bot_reply)
