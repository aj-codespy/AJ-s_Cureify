import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
import pickle
import os

model = SentenceTransformer('all-MiniLM-L6-v2')

def load_vector_db(index_path="faiss_index.idx", chunks_file="text_chunks.pkl"):
    index = faiss.read_index(index_path)
    with open(chunks_file, "rb") as f:
        text_chunks = pickle.load(f)
    return index, text_chunks

def get_text_embeddings(text):
    if not text.strip():
        return np.zeros(384)
    embeddings = model.encode([text])
    return embeddings

def answer_generation(input, chatHistory):
    llm = ChatGoogleGenerativeAI(
        model='gemini-1.5-flash',
        temperature=0,
        api_key=os.getenv("AIzaSyDtB4bETfNDyvpzA_NnBKMrr56rdiOE8bQ"),
        max_tokens=None,
        timeout=30,
        max_retries=2
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", '''You are a Medical AI that diagnoses diseases based on symptoms.
Instructions:
1. If symptoms clearly indicate a disease, provide a diagnosis.
2. If symptoms are unclear, ask a follow-up question.
3. If unsure, say: "I need more details."
4. Always ask **one** question at a time.
'''),
        MessagesPlaceholder("chat_history"),
        ("human", "{Question}")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"Question": input, "chat_history": chatHistory})
    chatHistory.extend([HumanMessage(content=input), response.content])
    return response.content

def query_vector_db(query_text, index, text_chunks, k=3):
    query_embedding = np.array(get_text_embeddings(query_text)).astype('float32').reshape(1, -1)
    distances, indices = index.search(query_embedding, k)
    retrieved_chunks = [text_chunks[i] for i in indices[0]]
    return "\n".join(retrieved_chunks)

def retrieve_and_answer(query_text, chatHistory, index_path="faiss_index.idx", chunks_file="text_chunks.pkl", k=4):
    index, stored_chunks = load_vector_db(index_path, chunks_file)
    context = query_vector_db(query_text, index, stored_chunks, k)
    
    while True:
        response = answer_generation(f"Context: {context}\nQuestion: {query_text}", chatHistory)
        if "diagnosis" in response.lower() or "i need more details" in response.lower():
            return response
        query_text = input("Follow-up question: ")
