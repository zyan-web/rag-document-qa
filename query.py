from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

llm = ChatOllama(model="llama3.2")

question = input("Apna sawaal pochain: ")

results = vectorstore.similarity_search(question, k=2)
context = "\n\n".join([doc.page_content for doc in results])

print("\n--- Retrieved Context ---")
print(context)
print("--------------------------\n")

prompt = f"""Only answer based on the context below. If the answer is not in the context, say "Sorry, I don't have information about this."

Context:
{context}

Question: {question}

Answer:"""

response = llm.invoke(prompt)
print("AI ka Jawab:")
print(response.content)
