"""Ollama API wrapper for chat and generate endpoints."""

import json
import requests


def ollama_chat(
    messages: list[dict],
    model: str,
    base_url: str = "http://10.0.0.16:11434",
    temperature: float = 0.7,
    timeout: int = 120,
) -> str:
    """Send a chat request to Ollama and return the assistant response text."""
    resp = requests.post(
        f"{base_url}/api/chat",
        json={
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def ollama_generate(
    prompt: str,
    model: str,
    base_url: str = "http://10.0.0.16:11434",
    temperature: float = 0.7,
    timeout: int = 120,
) -> str:
    """Send a generate request to Ollama and return the response text."""
    resp = requests.post(
        f"{base_url}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["response"]


def ollama_check(base_url: str = "http://10.0.0.16:11434") -> bool:
    """Check if Ollama is reachable."""
    try:
        resp = requests.get(f"{base_url}/api/tags", timeout=5)
        return resp.status_code == 200
    except requests.ConnectionError:
        return False


def ollama_list_models(base_url: str = "http://10.0.0.16:11434") -> list[str]:
    """List available Ollama models."""
    resp = requests.get(f"{base_url}/api/tags", timeout=10)
    resp.raise_for_status()
    return [m["name"] for m in resp.json().get("models", [])]
