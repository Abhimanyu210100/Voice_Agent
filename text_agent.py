from typing import List, Dict

import requests

_OLLAMA_URL = "http://localhost:11434/v1/chat/completions"
_MODEL = "llama3.1:8b"


def chat(system_prompt: str, history: List[Dict], user_message: str) -> str:
    """
    Send a candidate message and return the interviewer's reply.

    history: alternating {"role": "user"|"assistant", "content": str} entries
             representing the conversation so far (not including user_message).
    """
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    resp = requests.post(
        _OLLAMA_URL,
        headers={"Content-Type": "application/json"},
        json={
            "model": _MODEL,
            "messages": messages,
            "max_tokens": 500,
            "stream": False,
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()
