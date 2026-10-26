import os
from openai import OpenAI


def call_qwen(prompt: str) -> str:
    """
    Отправляет запрос к Qwen3.8-27B через OpenAI-совместимый API.
    """
    client = OpenAI()

    messages = [
        {"role": "system", "content": "Ты — экспертный ML-инженер. Отвечай строго в формате, указанном в запросе."},
        {"role": "user", "content": prompt},
    ]
    model = os.getenv("OPENAI_MODEL", "qwen/qwen3.8-27b")
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
            max_tokens=8192,
            extra_body={"enable_thinking": False},
        )
        return response.choices[0].message.content
    except Exception:
        # Фоллбэк, если extra_body не поддерживается
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
            max_tokens=8192,
        )
        return completion.choices[0].message.content