from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from typing import Dict, List, Optional
from dotenv import load_dotenv
import os
from datasets import Dataset
load_dotenv()
# RAGAS imports
try:
    from ragas import SingleTurnSample
    from ragas.metrics import BleuScore, NonLLMContextPrecisionWithReference, ResponseRelevancy, Faithfulness, RougeScore
    from ragas import evaluate
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False

def evaluate_response_quality(question: str, answer: str, contexts: List[str],
                            reference: Optional[str] = None,
                            enabled_metrics: Optional[List[str]] = None,
                            reference_contexts: Optional[List[str]] = None
                              ) -> Dict[str, float]:
    """Evaluate response quality using RAGAS metrics"""
    if not RAGAS_AVAILABLE:
        return {"error": "RAGAS not available"}
    
    # TODO: Create evaluator LLM with model gpt-3.5-turbo
    # TODO: Create evaluator_embeddings with model text-embedding-3-small
    # TODO: Define an instance for each metric to evaluate
    # TODO: Evaluate the response using the metrics
    # TODO: Return the evaluation results

    if enabled_metrics is None:
        enabled_metrics = [
            "response_relevancy",
            "faithfulness",
            "bleu",
            "rouge",
            "context_precision",
        ]

    try:
        # Evaluator LLM
        evaluator_llm = LangchainLLMWrapper(
            ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0,
                api_key=os.getenv("OPENAI_API_KEY")
            )
        )

        # Evaluator embeddings
        evaluator_embeddings = LangchainEmbeddingsWrapper(
            OpenAIEmbeddings(
                model="text-embedding-3-small",
                openai_api_key=os.getenv("OPENAI_API_KEY")
            )
        )



        # Create single-turn sample
        sample_data = {
            "user_input": question,
            "response": answer,
            "retrieved_contexts": contexts,
        }

        if reference:
            sample_data["reference"] = reference

        if reference_contexts:
            sample_data["reference_contexts"] = reference_contexts

        sample = SingleTurnSample(**sample_data)

        # Define metrics
        metrics = []

        if "response_relevancy" in enabled_metrics:
            metrics.append(
                ResponseRelevancy(
                    llm=evaluator_llm,
                    embeddings=evaluator_embeddings,
                )
            )

        if "faithfulness" in enabled_metrics:
            metrics.append(
                Faithfulness(
                    llm=evaluator_llm,
                )
            )

        # Reference-based metrics
        if reference:
            if "bleu" in enabled_metrics:
                metrics.append(BleuScore())

            if "rouge" in enabled_metrics:
                metrics.append(RougeScore())

            if "context_precision" in enabled_metrics  and reference_contexts:
                metrics.append(NonLLMContextPrecisionWithReference())
        else:
            skipped = [
                metric for metric in ["bleu", "rouge", "context_precision"]
                if metric in enabled_metrics
            ]
            if skipped:
                print(
                    f"Skipping reference-based metrics {skipped} because no reference was provided."
                )

        dataset = Dataset.from_list([sample.to_dict()])

        # Evaluate
        results = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=evaluator_llm,
            embeddings=evaluator_embeddings
        )

        # Convert results to dictionary
        scores = results.to_pandas().iloc[0].to_dict()

        return {
            key: float(value)
            for key, value in scores.items()
            if isinstance(value, (int, float))
        }

    except Exception as e:
        return {"error": str(e)}

    
