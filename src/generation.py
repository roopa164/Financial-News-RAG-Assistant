from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
# FIXED: RetrievalQA now lives in langchain_classic in v1.x
from langchain_classic.chains import RetrievalQA 
# FIXED: Base logic like prompts live in langchain_core
from langchain_core.prompts import PromptTemplate

def get_rag_chain(persist_dir):
    # Initialize the vector database
    vector_db = Chroma(
        persist_directory=persist_dir, 
        embedding_function=OpenAIEmbeddings()
    )

    # Use gpt-4o-mini with temperature 0 for deterministic, factual responses
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

    # Restricted RAG prompt to solve the 'Low Faithfulness' issue
    template = """
    You are a professional financial assistant. Answer the question based ONLY on the following context.
    If the answer is not contained within the context, say "I don't have enough information from the news to answer this."
    Do not add any outside information or personal opinions.

    Context:
    {context}

    Question: 
    {question}

    Answer:
    """
    
    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["context", "question"],
        template=template,
    )

    # Integrate the prompt into the RetrievalQA chain
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_db.as_retriever(search_kwargs={"k": 5}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT} 
    )
    
    print("RAG Chain initialized with restricted prompt.")
    return chain