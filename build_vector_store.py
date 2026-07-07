from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
import os
from langchain.embeddings import HuggingFaceEmbeddings

def load_chunks(folder="data_chunks"):
    docs = []
    for file in sorted(os.listdir(folder)):
        if file.endswith(".txt"):
            with open(os.path.join(folder, file), "r", encoding="utf-8") as f:
                docs.append(Document(page_content=f.read()))
    return docs

if __name__ == "__main__":
    docs = load_chunks()
    print(f"Loaded {len(docs)} chunks")

    model_name = "all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    print("Creating FAISS index with local embeddings...")
    vectorstore = FAISS.from_documents(docs, embeddings)

    vectorstore.save_local("faiss_index")
    print("Vector store saved locally.")
