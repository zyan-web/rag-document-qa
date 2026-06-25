from fastapi import FastAPI
from pydantic import BaseModel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Startup pe ek baar load hoga, har request pe nahi
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
llm = ChatOllama(model="llama3.2")

class Question(BaseModel):
    question: str

@app.get("/")
def home():
    return {"message": "RAG API is running!"}

@app.post("/ask")
def ask(body: Question):
    results = vectorstore.similarity_search(body.question, k=2)
    context = "\n\n".join([doc.page_content for doc in results])
    
    prompt = f"""Only answer based on the context below. If answer is not in context, say "Sorry, I don't have information about this."

Context:
{context}

Question: {body.question}

Answer:"""
    
    response = llm.invoke(prompt)
    return {"answer": response.content, "context_used": context}

