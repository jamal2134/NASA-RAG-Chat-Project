from typing import Dict, List
from openai import OpenAI

def generate_response(
    openai_key: str,
    user_message: str,
    context: str,
    conversation_history: List[Dict],
    model: str = "gpt-3.5-turbo"
) -> str:
    """Generate response using OpenAI with context"""

    system_prompt = """
You are a NASA mission archive Q&A assistant.

Your task is to answer questions about historic NASA missions, especially Apollo 11, Apollo 13, and Challenger, using only the provided context documents.

Rules:
- Answer based only on the provided context.
- If the context does not contain enough information, clearly say that.
- Do not invent facts.
- Give accurate, clear, and detailed answers.
- When possible, mention which mission or document the information comes from.
- Keep the answer professional and easy to understand.
"""

    user_prompt = f"""
Based on the following context documents, answer the user's question.

Context Documents:
{context}

User Question:
{user_message}

Please provide a comprehensive answer based only on the context provided.
"""

    messages = []

    # Add system prompt first
    messages.append({
        "role": "system",
        "content": system_prompt
    })

    # Add previous conversation history without old system prompts
    for msg in conversation_history:
        if msg.get("role") != "system":
            messages.append(msg)

    # Add current user question with retrieved context
    messages.append({
        "role": "user",
        "content": user_prompt
    })

    # Create OpenAI client
    openai_client = OpenAI(api_key=openai_key)

    # Send request
    response = openai_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2, # We use low temperature like 0.2 because you want factual answers, not creative answers.
        max_tokens=500
    )

    return response.choices[0].message.content