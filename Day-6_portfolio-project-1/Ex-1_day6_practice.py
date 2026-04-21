import json
import os
from typing import Any, Dict, List, Optional

import requests


def analyze_reviews_sentiment(
    reviews: List[str],
    api_key: Optional[str] = None,
    model: str = "llama-3.1-8b-instant",
) -> List[Dict[str, Any]]:
    """
    Analyze a list of product reviews with the Groq API.

    Returns a list of dicts. Each dict includes:
    - review: original review text
    - sentiment: positive | negative | neutral (when available)
    - confidence: confidence score from model output (when available)
    - reason: short explanation from model output (when available)
    - error: error details (only when analysis failed)
    """
    if not isinstance(reviews, list):
        raise TypeError("reviews must be a list of strings")

    resolved_api_key = api_key or os.getenv("GROQ_API_KEY")
    if not resolved_api_key:
        raise ValueError("Groq API key is required (pass api_key or set GROQ_API_KEY).")

    endpoint = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {resolved_api_key}",
        "Content-Type": "application/json",
    }

    results: List[Dict[str, Any]] = []

    for review in reviews:
        result: Dict[str, Any] = {"review": review}

        if not isinstance(review, str) or not review.strip():
            result["error"] = "Review must be a non-empty string."
            results.append(result)
            continue

        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a sentiment analysis assistant. "
                        "Return ONLY valid JSON with keys: sentiment, confidence, reason. "
                        "sentiment must be one of: positive, negative, neutral."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Analyze this product review:\n{review}",
                },
            ],
            "temperature": 0,
        }

        try:
            response = requests.post(endpoint, headers=headers, json=payload, timeout=30)
            response.raise_for_status()

            data = response.json()
            content = data["choices"][0]["message"]["content"].strip()

            try:
                parsed = json.loads(content)
                result["sentiment"] = parsed.get("sentiment")
                result["confidence"] = parsed.get("confidence")
                result["reason"] = parsed.get("reason")
            except json.JSONDecodeError:
                # Fallback if model does not return strict JSON.
                result["sentiment"] = None
                result["confidence"] = None
                result["reason"] = content
                result["error"] = "Model response was not valid JSON."

        except requests.exceptions.Timeout:
            result["error"] = "Request timed out while contacting Groq API."
        except requests.exceptions.HTTPError as exc:
            result["error"] = f"HTTP error from Groq API: {exc}"
        except requests.exceptions.RequestException as exc:
            result["error"] = f"Network error while contacting Groq API: {exc}"
        except (KeyError, IndexError, ValueError) as exc:
            result["error"] = f"Unexpected Groq API response format: {exc}"

        results.append(result)

    return results
