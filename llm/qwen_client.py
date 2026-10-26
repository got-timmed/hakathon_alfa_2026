import os
from openai import OpenAI


def call_qwen(prompt: str, model: str = "qwen3.8-27b") -> str:
    """
    Отправляет запрос к Qwen3.8-27B через OpenAI-совместимый API.
    """
    api_key = os.getenv("QWEN_API_KEY")
    if not api_key:
        raise ValueError("Не задан QWEN_API_KEY. Проверь переменные окружения.")

    base_url = os.getenv("QWEN_BASE_URL", "https://ap-southeast-1.maas.qwencloudapi.com/compatible-mode/v1")

    client = OpenAI(api_key=api_key, base_url=base_url)

    messages = [
        {"role": "system", "content": "Ты — экспертный ML-инженер. Отвечай строго в формате, указанном в запросе."},
        {"role": "user", "content": prompt},
    ]

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
            max_tokens=8192,
            extra_body={"enable_thinking": False},
        )
        return completion.choices[0].message.content
    except Exception:
        # Фоллбэк, если extra_body не поддерживается
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
            max_tokens=8192,
        )
        return completion.choices[0].message.content