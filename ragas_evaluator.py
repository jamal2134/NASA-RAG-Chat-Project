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

def evaluate_response_quality(question: str, answer: str, contexts: List[str]) -> Dict[str, float]:
    """Evaluate response quality using RAGAS metrics"""
    if not RAGAS_AVAILABLE:
        return {"error": "RAGAS not available"}
    
    # TODO: Create evaluator LLM with model gpt-3.5-turbo
    # TODO: Create evaluator_embeddings with model text-embedding-3-small
    # TODO: Define an instance for each metric to evaluate
    # TODO: Evaluate the response using the metrics
    # TODO: Return the evaluation results

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
        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts
        )

        # Define metrics
        metrics = [
            ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
            Faithfulness(llm=evaluator_llm)
        ]

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

    
