import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# .env file se API keys load karo
load_dotenv()

# Step 1: Document load karo
loader = TextLoader("company_info.txt")
documents = loader.load()
print(f"Document loaded: {len(documents)} document(s)")

# Step 2: Chunks mein todo
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20
)
chunks = text_splitter.split_documents(documents)
print(f"Document split into {len(chunks)} chunks")

# Step 3: Embeddings banao (free, local, Hugging Face model) aur ChromaDB mein store karo
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)
print("Chunks stored in ChromaDB successfully!")
