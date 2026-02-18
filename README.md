🗞️ Financial News RAG Assistant

An AI-powered Retrieval-Augmented Generation (RAG) system designed to provide factual, context-grounded answers about financial news. This project leverages a modular architecture with a FastAPI backend for heavy-lifting AI logic and a Streamlit frontend for a clean user interface.

🏗️ Project Architecture

The system is divided into two main components to ensure scalability and clean separation of concerns:

  - Backend (FastAPI): Exposes the RAG pipeline via a REST API. It handles query embedding, searching the Chroma vector database, and generating answers using gpt-4o-mini.

  - Frontend (Streamlit): A web dashboard where users can submit questions and view the AI's answer alongside the specific news snippets used as evidence.


🔍 Observability & Monitoring

Production-ready AI systems require visibility beyond just functionality. This project integrates LLM tracing and structured logging to ensure debuggability, transparency, and operational readiness.

 1️⃣ LangSmith Tracing (LLM-Level Observability)

    The RAG pipeline is instrumented with LangSmith to trace:

    Retrieval inputs and outputs

    Prompt construction

    LLM responses

    Token usage

    Latency per run

Each user query is logged as a trace, allowing inspection of:

    Retrieved documents

    Final composed prompt

    Model response

    Execution time

This makes it possible to debug hallucinations, diagnose retrieval failures, and optimize performance.

To enable tracing, add the following to your .env:




📁 Folder Structure
Plaintext

AI_Project_2026/

      ├── src/                # Core logic (Brain)
      │   ├── ingestion.py    # JSON Data -> ChromaDB
      │   └── generation.py   # RAG Chain logic
      ├── app.py              # FastAPI Server (The Engine)
      ├── ui.py               # Streamlit App (The Face)
      ├── chroma_db/          # Vector Database storage
      ├── .env                # API Keys (Excluded from Git)
      └── requirements.txt    # Project dependencies
🛠️ Installation & Setup


    Bash
    
    pip install -r requirements.txt
Configure Environment Variables: Create a .env file in the root directory and add your OpenAI key:

Plaintext

    OPENAI_API_KEY=sk-xxxx...
🚀 How to Run

You need to run the backend and frontend in separate terminal windows.

Step 1: Start the FastAPI Backend
      
      uvicorn app:app --reload
The API will be available at http://127.0.0.1:8000. You can view the interactive API docs at /docs.

Step 2: Start the Streamlit Frontend
   
    
    streamlit run ui.py
The dashboard will open automatically in your browser.

📊 Evaluation & Metrics

This project uses the Ragas framework to measure performance across:

    Faithfulness: Ensuring the AI doesn't hallucinate.

    Answer Relevance: Checking if the answer actually addresses the user's query.

    Context Precision: Measuring how "noisy" the retrieved news snippets are.

💡 Implementation Note

This project uses langchain-classic to maintain compatibility with the RetrievalQA chain while benefiting from the speed of the 2026 LangChain v1.x core.
