from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import create_retriever_tool
import os
import pandas as pd
from langchain_core.vectorstores import InMemoryVectorStore
import langchain

loader1 = PyPDFLoader("IEEE-Test-Doc-829-2008.pdf")
loader2 = PyPDFLoader("29119-1-2022.pdf")

raw_pdf_pages1 = loader1.load()
raw_pdf_pages2 = loader2.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)

document1_chunks = text_splitter.split_documents(raw_pdf_pages1)
document2_chunks = text_splitter.split_documents(raw_pdf_pages2)


embeddings = OllamaEmbeddings(model="nomic-embed-text")

vector_store = Chroma(
    collection_name= "iso_files",
    embedding_function= embeddings,
    persist_directory= "./chromadb"
)
if not vector_store.get()['ids']:
    vector_store.add_documents(documents=document1_chunks + document2_chunks)

retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 5})

retriever_tool = create_retriever_tool(
    retriever,
    name="search_testing_standards",
    description="Search for software testing standards, requirements, and guidelines from IEEE 829 and ISO 29119. Always use this tool before generating test cases."
)
tools = [retriever_tool]
