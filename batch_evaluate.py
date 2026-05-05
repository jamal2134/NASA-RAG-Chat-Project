import os
import json
import pandas as pd
from dotenv import load_dotenv

from rag_client import initialize_rag_system, retrieve_documents, format_context
from llm_client import generate_response
from ragas_evaluator import evaluate_response_quality

load_dotenv()


def run_batch_evaluation(
    test_file="test_questions.json",
    chroma_dir="./chroma_db_openai",
    collection_name="nasa_space_missions_text",
    n_results=3,
    output_file="batch_evaluation_results.csv"
):

    print("-----------------------------------------------------------")
    print(f"File: {test_file}")
    print("-----------------------------------------------------------")
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError("OPENAI_API_KEY is missing. Add it to .env or environment variables.")

    collection, success, error = initialize_rag_system(
        chroma_dir,
        collection_name
    )

    if not success:
        raise RuntimeError(f"Failed to initialize RAG system: {error}")

    with open(test_file, "r", encoding="utf-8") as f:
        test_questions = json.load(f)

    results = []

    for item in test_questions:
        question = item["question"]
        category = item.get("category", "general")

        print(f"\nEvaluating: {question}")

        retrieved = retrieve_documents(
            collection=collection,
            query=question,
            n_results=n_results
        )

        if not retrieved or not retrieved.get("documents"):
            results.append({
                "category": category,
                "question": question,
                "answer": "",
                "answer_relevancy": None,
                "faithfulness": None,
                "error": "No documents retrieved"
            })
            continue

        documents = retrieved["documents"][0]
        metadatas = retrieved["metadatas"][0]

        context = format_context(documents, metadatas)

        answer = generate_response(
            openai_key=api_key,
            user_message=question,
            context=context,
            conversation_history=[]
        )

        reference = item.get("reference")



        scores = evaluate_response_quality(
            question=question,
            answer=answer,
            contexts=documents,
            reference=reference,
            reference_contexts=documents,
            enabled_metrics=[
                "response_relevancy",
                "faithfulness",
                "bleu",
                "rouge",
                "context_precision"
            ]
        )

        row = {
            "category": category,
            "question": question,
            "answer": answer,
            "error": scores.get("error", "")
        }

        for metric_name, value in scores.items():
            if metric_name != "error":
                row[metric_name] = value

        results.append(row)

    df = pd.DataFrame(results)

    print("\n==============================")
    print("Per Question Evaluation Summary")
    print("==============================")
    print(df[["category", "question", "answer_relevancy", "faithfulness", "error"]])

    print("\n==============================")
    print("Aggregate Metrics")
    print("==============================")

    numeric_cols = df.select_dtypes(include="number").columns

    for col in numeric_cols:
        print(f"Mean {col}: {df[col].mean():.3f}")

    df.to_csv(output_file, index=False, encoding="utf-8-sig")

    print(f"\nResults saved to: {output_file}")

    return df


if __name__ == "__main__":
    ## General Batch Evaluation
    run_batch_evaluation()
    ## Batch Evaluation Dataset
    run_batch_evaluation(test_file="test_questions_with_references.json",output_file="batch_evaluation_results_with_references.csv")