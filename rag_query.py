# rag_query.py
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

INDEX_DIR = "rag_index"

PROMPT = ChatPromptTemplate.from_template("""
You are a concise, accurate assistant. Use the provided context to answer the user's question.
If the answer isn't in the context, say you don't know.

Question:
{question}

Context:
{context}
""".strip())

def format_docs(docs):
    return "\n\n".join(f"[{i+1}] {d.page_content}" for i, d in enumerate(docs))

def build_chain():
    # Local models via Ollama
    llm = ChatOllama(model="Qwen2.5:0.5B", temperature=0.2)
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    # Load FAISS index
    vs = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)

    # Retriever
    retriever = vs.as_retriever(search_type="mmr", search_kwargs={"k": 4})

    # Chain: retrieve → prompt → LLM → text
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | PROMPT
        | llm
        | StrOutputParser()
    )
    return chain

def ask(question: str):
    chain = build_chain()
    print(chain.invoke(question))

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python rag_query.py \"your question\"")
        sys.exit(1)
    ask(" ".join(sys.argv[1:]))
