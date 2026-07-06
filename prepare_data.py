import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import PyPDF2

def load_pdf(file_path):
    text = ""
    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

def chunk_text(text, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_text(text)
    return [Document(page_content=chunk) for chunk in chunks]

if __name__ == "__main__":
    full_text = load_pdf("Cheezious-data.pdf")
    print(f"Loaded text length: {len(full_text)} characters")

    docs = chunk_text(full_text)
    print(f"Split into {len(docs)} chunks.")

    if not os.path.exists("data_chunks"):
        os.mkdir("data_chunks")
    for i, doc in enumerate(docs):
        with open(f"data_chunks/chunk_{i}.txt", "w", encoding="utf-8") as f:
            f.write(doc.page_content)

    print("Chunks saved to data_chunks folder.")
