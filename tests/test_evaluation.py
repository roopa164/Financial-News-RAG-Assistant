import pandas as pd
import os
from datasets import Dataset
from ragas import evaluate
from dotenv import load_dotenv

# Use the latest Capitalized Metric classes from Ragas v0.4+
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    AnswerCorrectness
)
from src.generation import get_rag_chain

load_dotenv()

def test_offline_rag_evaluation():
    # 1. Initialize your actual RAG chain
    chain = get_rag_chain("chroma_db")

    # 2. Load your Ground Truth CSV
    # Ensure this path matches your file location
    csv_path = "tests/ground_truth.csv"
    df_gt = pd.read_csv(csv_path)

    results = []
    
    print(f"--- Starting evaluation for {len(df_gt)} questions ---")

    # 3. Generate live responses for each question
    for index, row in df_gt.iterrows():
        # Mapping from your specific CSV headers
        query = row['Question']
        ground_truth = row['Ground Truth (Reference)']
        
        print(f"Evaluating Q{index+1}: {query[:50]}...")
        
        # Get live answer and contexts from your RAG pipeline
        response = chain.invoke({"query": query})
        
        # USE THE NEW RAGAS SCHEMA NAMES HERE
        results.append({
            "user_input": query,             # New standard for 'question'
            "response": response["result"],  # New standard for 'answer'
            "retrieved_contexts": [doc.page_content for doc in response["source_documents"]], # For 'contexts'
            "reference": ground_truth        # New standard for 'ground_truth'
        })

    # 4. Prepare the Ragas Dataset
    eval_dataset = Dataset.from_pandas(pd.DataFrame(results))

    # 5. Run the evaluation
    # Initializing metrics as classes with () to ensure they use default configurations
    score = evaluate(
        eval_dataset,
        metrics=[
            Faithfulness(),
            AnswerRelevancy(),
            ContextPrecision(),
            ContextRecall(),
            AnswerCorrectness()
        ]
    )

    # 6. Output and Export Results
    print("\n" + "="*30)
    print("FINAL RAGAS SCORES")
    print("="*30)
    print(score)
    
    # Save results for offline review
    df_scores = score.to_pandas()
    df_scores.to_csv("evaluation_results_detailed.csv", index=False)
    print("\nDetailed results saved to 'evaluation_results_detailed.csv'")

    # Updated assertion check for Faithfulness
    #assert score['faithfulness'] > 0.6