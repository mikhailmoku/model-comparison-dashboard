"""
comparator.py
Ядро проекта: отправляет один и тот же промпт нескольким моделям через Ollama
и замеряет скорость ответа (time-to-first-token, токенов/сек, общее время)
"""

import time
import requests
from dataclasses import dataclass, field
from typing import Optional


OLLAMA_URL = "http://localhost:11434/api/generate"


@dataclass
class ModelResult:
    model: str
    prompt: str
    response: str = ""
    error: Optional[str] = None
    total_time_sec: float = 0.0
    tokens_generated: int = 0
    tokens_per_sec: float = 0.0
    time_to_first_token_sec: Optional[float] = None


def run_prompt(model_name: str, prompt: str, stream: bool = True, timeout: int = 120) -> ModelResult:
    """
    Отправляет промпт одной модели через Ollama и замеряет метрики скорости
    Если stream=True  считает time-to-first-token (более честная метрика отзывчивости)
    """
    result = ModelResult(model=model_name, prompt=prompt)
    payload = {"model": model_name, "prompt": prompt, "stream": stream}

    start = time.perf_counter()
    first_token_time = None
    full_text = []

    try:
        if stream:
            with requests.post(OLLAMA_URL, json=payload, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if not line:
                        continue
                    import json as _json
                    chunk = _json.loads(line)
                    token = chunk.get("response", "")
                    if token and first_token_time is None:
                        first_token_time = time.perf_counter() - start
                    full_text.append(token)
                    if chunk.get("done"):
                        # Ollama возвращает eval_count (кол-во сгенерированных токенов) в финальном чанке
                        result.tokens_generated = chunk.get("eval_count", 0)
            result.response = "".join(full_text)
        else:
            resp = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            result.response = data.get("response", "")
            result.tokens_generated = data.get("eval_count", 0)

        result.total_time_sec = time.perf_counter() - start
        result.time_to_first_token_sec = first_token_time
        if result.total_time_sec > 0 and result.tokens_generated:
            result.tokens_per_sec = result.tokens_generated / result.total_time_sec

    except requests.exceptions.ConnectionError:
        result.error = "Не удалось подключиться к Ollama. Убедитесь, что сервис запущен (ollama serve)"
    except requests.exceptions.Timeout:
        result.error = f"Модель не ответила за {timeout} секунд"
    except Exception as e:
        result.error = f"Ошибка: {e}"

    return result


def compare_models(prompt: str, models: list[str], stream: bool = True) -> list[ModelResult]:
    """Прогоняет один промпт через список моделей и возвращает список результатов"""
    results = []
    for model in models:
        results.append(run_prompt(model, prompt, stream=stream))
    return results

