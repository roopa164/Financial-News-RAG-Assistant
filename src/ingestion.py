import json
import os
from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

def load_and_process_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    documents = []
    # Loop through each ticker and its list of news articles
    for ticker, articles in data.items():
        for article in articles:
            # Create a LangChain Document with metadata for better retrieval
            doc = Document(
                page_content=article['full_text'],
                metadata={
                    "ticker": ticker,
                    "title": article['title'],
                    "link": article['link']
                }
            )
            documents.append(doc)
    return documents

def build_vector_store(json_file, persist_dir):
    # 1. Load data from your stock_news.json
    raw_docs = load_and_process_json(json_file)

    # 2. Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(raw_docs)

    # 3. Create Vector DB
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=OpenAIEmbeddings(),
        persist_directory=persist_dir
    )
    print(f"Success: Processed {len(chunks)} chunks from {json_file}")
    return vector_db


# JSON_DATA = os.path.join("data", "stock_news.json")
# DB_DIR = "./chroma_db"
# build_vector_store(JSON_DATA, DB_DIR)