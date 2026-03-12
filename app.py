from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from transformers import pipeline

loader = PyPDFLoader("document.pdf")
documents = loader.load()

embeddings = HuggingFaceEmbeddings()

vector_store = FAISS.from_documents(documents, embeddings)

retriever = vector_store.as_retriever()

qa = RetrievalQA.from_chain_type(
    llm=pipeline("text-generation"),
    retriever=retriever
)

print(qa.run("What is this document about?"))
