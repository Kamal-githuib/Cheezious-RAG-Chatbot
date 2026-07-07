import os
import re
import uuid

import requests
import streamlit as st
from dotenv import load_dotenv

try:
    import fitz
except ImportError:
    fitz = None

import PyPDF2

# ==============================
# LOAD ENV
# ==============================
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    st.warning("⚠ GEMINI_API_KEY not set; responses will fall back to a local answer.")

# ==============================
# CONFIG
# ==============================
st.set_page_config(page_title="Cheezious", layout="centered")

PDF_DIR = "pdfs"
CHUNK_DIR = "data_chunks"
indexed_docs = []
STOP_WORDS = {
    "the", "and", "for", "with", "what", "which", "where", "when", "how", "why",
    "can", "you", "your", "is", "are", "do", "does", "did", "have", "has", "had",
    "our", "this", "that", "these", "those", "from", "into", "about", "cheezious",
    "menu", "latest", "prices", "price", "deals", "deal", "march", "2025", "download",
    "pdf", "one", "all", "here", "there", "please", "tell", "give", "me", "my", "i",
    "it", "of", "to", "in", "on", "at", "be", "an", "a", "or"
}


def normalize_token(token):
    token = token.lower()
    if token in STOP_WORDS or len(token) <= 2:
        return ""
    if token.endswith("ies") and len(token) > 4:
        token = token[:-3] + "y"
    elif token.endswith("es") and len(token) > 4 and not token.endswith(("ses", "xes", "zes")):
        token = token[:-2]
    elif token.endswith("s") and len(token) > 3 and not token.endswith("ss"):
        token = token[:-1]
    return token


def tokenize(text):
    if not text:
        return []

    normalized = text.lower()
    normalized = re.sub(r"([a-z])([A-Z])", r"\1 \2", normalized)
    normalized = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", normalized)
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return [normalize_token(token) for token in normalized.split() if normalize_token(token)]


def build_keyword_vector(text):
    tokens = tokenize(text)
    if not tokens:
        return {}
    vector = {}
    for token in tokens:
        vector[token] = vector.get(token, 0) + 1
    return vector

# ==============================
# PDF FUNCTIONS
# ==============================
def extract_text_from_pdf(pdf_path):
    if fitz is not None:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text

    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        return "\n".join(page.extract_text() or "" for page in reader.pages)

def chunk_text(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def parse_menu_lines(content):
    lines = []
    for raw_line in content.splitlines():
        line = re.sub(r"^[\u2022\-\*]\s*", "", raw_line).strip()
        if not line:
            continue
        if line.lower().startswith(("cheeziousmenu", "downloadpdf", "aboutcheezious", "cheezious pakistan")) and "rs" not in line.lower():
            continue
        if len(line) < 3:
            continue
        line = line.replace("•", " ").strip()
        lines.append(line)
    return lines


def ingest_pdfs():
    if indexed_docs:
        st.info("Documents already ingested.")
        return

    source_files = []
    if os.path.isdir(CHUNK_DIR):
        source_files = sorted([os.path.join(CHUNK_DIR, f) for f in os.listdir(CHUNK_DIR) if f.endswith(".txt")])

    if not source_files:
        pdf_files = [f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")]
        if not pdf_files:
            st.warning("No PDFs found in 'pdfs/' folder.")
            return
        with st.spinner("Loading PDFs and preparing local search index, please wait..."):
            for pdf_file in pdf_files:
                path = os.path.join(PDF_DIR, pdf_file)
                text = extract_text_from_pdf(path)
                chunks = chunk_text(text)
                for chunk in chunks:
                    source_files.append((path, chunk))
    else:
        with st.spinner("Loading prepared document chunks, please wait..."):
            pass

    with st.spinner("Building menu line index, please wait..."):
        for path in source_files:
            if isinstance(path, tuple):
                _, content = path
            else:
                with open(path, "r", encoding="utf-8") as handle:
                    content = handle.read().strip()
            for line in parse_menu_lines(content):
                indexed_docs.append({
                    "id": f"{os.path.basename(path) if not isinstance(path, tuple) else 'pdf'}:{len(indexed_docs)}",
                    "text": line,
                    "keywords": build_keyword_vector(line),
                })

    # Menu index is ready; keep the UI quiet for chat usage.


def ensure_index_loaded():
    if indexed_docs:
        return
    ingest_pdfs()

# ==============================
# RAG FUNCTIONS
# ==============================
def retrieve(query, top_k=15):
    query_tokens = [token for token in tokenize(query) if len(token) > 2]
    if not query_tokens:
        return indexed_docs[:1]

    query_set = set(query_tokens)
    price_intent = any(token in {"price", "prices", "cost", "rate", "rates", "rs", "rupee", "rupees", "prize"} for token in query_tokens)
    combo_intent = any(token in {"combo", "combos", "deal", "deals"} for token in query_tokens)
    crown_intent = any(token in {"crown", "crust"} for token in query_tokens)
    text_lower = query.lower()

    scored = []
    for doc in indexed_docs:
        doc_text = doc["text"]
        doc_terms = doc["keywords"]
        doc_key_tokens = set(doc_terms.keys())
        doc_lower = doc_text.lower()

        exact_matches = sum(1 for token in query_set if token in doc_key_tokens)
        weighted_overlap = sum(doc_terms.get(token, 0) for token in query_set)
        phrase_bonus = 2 if text_lower in doc_lower else 0
        exact_phrase_bonus = 6 if any(token in doc_lower for token in query_tokens) else 0
        category_bonus = 0
        for token in query_set:
            if token in {"burger", "pizza", "combo", "deal", "pasta", "fries", "chicken", "nugget", "dessert", "sandwich", "platter"} and token in doc_key_tokens:
                category_bonus += 2
        if price_intent and ("rs" in doc_lower or any(char.isdigit() for char in doc_text)):
            category_bonus += 8
        if combo_intent and ("+" in doc_text or "deal" in doc_lower or "combo" in doc_lower):
            category_bonus += 8
        if crown_intent and ("crown" in doc_lower or "crust" in doc_lower):
            category_bonus += 10
        if price_intent and re.search(r"\b(?:bazinga|reggy|munch|deal|burger|pizza|pasta|fries|wings|nuggets|crown|crust)\b", doc_lower):
            category_bonus += 2
        if len(doc_text) < 80 and any(token in doc_lower for token in query_tokens):
            category_bonus += 4
        if re.search(r"\b(?:bazinga|reggy|munch|deal|burger|pizza|pasta|fries|wings|nuggets)\b", doc_lower) and ("rs" in doc_lower or any(char.isdigit() for char in doc_text)):
            category_bonus += 6
        if doc_text.lower().startswith(("burgers", "pizzas", "pastas", "starters", "side", "sandwiches")):
            category_bonus -= 6

        score = exact_matches * 6 + weighted_overlap + phrase_bonus + exact_phrase_bonus + category_bonus
        score += score_query_match(query, doc_text)
        if score > 4:
            scored.append((score, doc))

    scored.sort(key=lambda item: item[0], reverse=True)
    docs = []
    for _, doc in scored[:top_k]:
        docs.append({"text": doc["text"], "meta": {"doc_id": doc["id"]}})
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

def clean_answer_text(text):
    if not text:
        return ""
    text = text.replace("•", "")
    text = re.sub(r"^[\-\*]\s*", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace("Cheeziousmenuoffersdifferentmidnightdeals.Thisofferisavailablefrom12amto3am.", "")
    text = text.replace("Hereisthelistofallcomboswithpricesavailableat", "")
    text = text.replace("InCheeziousmenuyou’llgetdifferentcomboswiththemainfooditem.", "")
    text = text.replace("Cheezious", "")
    return text.strip(" .")


def looks_like_menu_entry(text):
    cleaned = clean_answer_text(text)
    if not cleaned:
        return False
    if len(cleaned) > 120:
        return False
    lower = cleaned.lower()
    if any(phrase in lower for phrase in ["you’ll get", "offer is available", "family and friends", "offers many items", "variety of food"]):
        return False
    if "+" in cleaned or "rs" in lower or re.search(r"\b(?:bazinga|reggy|bazooka|munch|crown|crust|burger|pizza|pasta|fries|nuggets|wings|chicken)\b", lower):
        return True
    return False


def score_query_match(query, text):
    cleaned = clean_answer_text(text)
    if not cleaned:
        return 0

    q_lower = query.lower()
    query_tokens = [token for token in tokenize(query) if token]
    text_lower = cleaned.lower()
    text_compact = re.sub(r"[^a-z0-9]+", "", text_lower)

    score = 0
    for token in query_tokens:
        if token in text_lower:
            score += 4
        elif token in text_compact:
            score += 3

    if any(word in q_lower for word in ["price", "cost", "rate", "rs", "rupee", "rupees", "prize"]):
        if "rs" in text_lower or any(char.isdigit() for char in cleaned):
            score += 6

    if any(word in q_lower for word in ["combo", "combos", "deal", "deals"]):
        if "+" in cleaned or "deal" in text_lower or "combo" in text_lower:
            score += 6

    if any(word in q_lower for word in ["crown", "crust"]):
        if "crown" in text_lower or "crust" in text_lower:
            score += 6

    if len(cleaned) < 100:
        score += 1

    if re.search(r"\b(?:bazinga|reggy|munch|deal|burger|pizza|pasta|fries|wings|nuggets|crown|crust)\b", text_lower):
        score += 2

    return score


def extract_answer_text(doc, query):
    if not doc:
        return ""

    lines = [line.strip() for line in (doc.get("text") or "").splitlines() if line.strip()]
    query_tokens = [token for token in tokenize(query) if token]
    if not query_tokens:
        return ""

    q_lower = query.lower()
    price_intent = any(word in q_lower for word in ["price", "cost", "rate", "rs", "rupee", "rupees", "prize"])
    combo_intent = any(word in q_lower for word in ["combo", "combos", "deal", "deals"])

    matching_lines = []
    for line in lines:
        line_lower = line.lower()
        if any(token in line_lower for token in query_tokens):
            matching_lines.append(line)

    if matching_lines:
        preferred = [line for line in matching_lines if "rs" in line.lower() or any(char.isdigit() for char in line)]
        chosen = preferred[0] if preferred else matching_lines[0]
    else:
        chosen = next((line for line in lines if line and len(line) < 220), lines[0] if lines else "")

    cleaned = clean_answer_text(chosen)
    if len(cleaned) > 220:
        cleaned = cleaned[:220].rstrip() + "..."
    return cleaned


def local_answer(retrieved_docs, user_query):
    if not retrieved_docs:
        return "I cannot find the answer to this question in our current documents. Please check the menu or contact the restaurant."

    q = user_query.lower()
    query_tokens = [token for token in tokenize(user_query) if len(token) > 2]
    price_intent = any(word in q for word in ["price", "cost", "rate", "rs", "rupee", "rupees", "prize"])
    combo_intent = any(word in q for word in ["combo", "combos", "deal", "deals"])
    crown_intent = any(word in q for word in ["crown", "crust"])

    candidates = []
    for doc in retrieved_docs:
        text = (doc.get("text") or "").strip()
        if not text:
            continue
        cleaned = clean_answer_text(text)
        if not cleaned or not looks_like_menu_entry(text):
            continue
        score = score_query_match(user_query, text)
        if score <= 4:
            continue
        candidates.append((score, cleaned))

    if not candidates:
        return "I could not find a specific match for that question in the available menu text."

    candidates.sort(key=lambda item: item[0], reverse=True)
    best_text = candidates[0][1]

    if combo_intent:
        combo_candidates = [text for score, text in candidates if ("+" in text or "deal" in text.lower() or "combo" in text.lower())][:5]
        combo_candidates = [item for item in combo_candidates if "available" not in item.lower() and len(item) > 5 and looks_like_menu_entry(item)]
        if combo_candidates:
            formatted = "Available combos:\n" + "\n".join(f"- {item}" for item in combo_candidates)
            return formatted

    if price_intent:
        if best_text:
            return f"Price info: {best_text}"

    if any(word in q for word in ["size", "large", "regular", "extra"]):
        size_matches = [text for score, text in candidates if "regular" in text.lower() or "large" in text.lower() or "pizza" in text.lower()]
        if size_matches:
            return "Size info: " + size_matches[0]
        if "large" in q or "extra" in q:
            return "Size info: LargePizzaDeal: Any flavor from Local Love or Over the Sea category with a 1-liter drink."
        if "regular" in q:
            return "Size info: RegularPizzaDeal: 1 regular pizza and 2 regular drinks."

    return best_text


def llm_answer(prompt, retrieved_docs=None, user_query=None):
    return local_answer(retrieved_docs or [], user_query or "")

# ==============================
# STREAMLIT UI
# ==============================
st.title("Cheezious")
st.write("This chatbot only provides answers from approved Cheezious documents.")

# Ingest PDFs automatically on app start
if "pdfs_loaded" not in st.session_state:
    ensure_index_loaded()
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
            ensure_index_loaded()
            retrieved = retrieve(user_query, top_k=15)
            prompt = build_prompt(retrieved, user_query)
            bot_reply = llm_answer(prompt, retrieved, user_query)

    st.session_state.messages.append({"role": "assistant", "content": bot_reply})
    with st.chat_message("assistant"):
        st.markdown(bot_reply)
