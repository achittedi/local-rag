# ingest.py
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

DOCS_DIR = "docs"
INDEX_DIR = "rag_index"

def load_docs():
    txt_loader = DirectoryLoader(
        "docs",
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={
            "autodetect_encoding": True,  # tries chardet if installed
            "encoding": "utf-8",          # fallback/default
        },
    )
    # add other loaders (PDF, MD, etc.) as needed…
    return txt_loader.load()

def main():
    print("Loading documents...")
    docs = load_docs()
    print(f"Loaded {len(docs)} documents")

    print("Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=150, add_start_index=True
    )
    chunks = splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks")

    print("Computing embeddings (local via Ollama)...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    print("Building FAISS index...")
    vs = FAISS.from_documents(chunks, embedding=embeddings)

    Path(INDEX_DIR).mkdir(exist_ok=True)
    vs.save_local(INDEX_DIR)
    print(f"Saved index to: {INDEX_DIR}")

if __name__ == "__main__":
    main()
