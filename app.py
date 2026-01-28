from fastapi import FastAPI
from pydantic import BaseModel
from src.generation import get_rag_chain

from dotenv import load_dotenv
load_dotenv()
# 1. Initialize the Engine
app = FastAPI(title="RAG Financial API")
chain = get_rag_chain("chroma_db")

# Define what a "Request" looks like
class QueryRequest(BaseModel):
    question: str

@app.post("/ask")
async def ask_rag(request: QueryRequest):
    # 2. Run the RAG logic
    response = chain.invoke({"query": request.question})
    
    # 3. Return a standardized package of data
    return {
        "answer": response["result"],
        "sources": [doc.page_content for doc in response["source_documents"]]
    }

# To run: uvicorn app:app --reload