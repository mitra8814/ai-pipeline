import requests
import json
import re
from typing import List, Optional
from loguru import logger

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "phi3:mini"


def _extract_json_array(content: str) -> Optional[List]:
    """Extract the first valid JSON array from content, even if followed by extra text."""
    content = content.strip()
    
    # Try to extract JSON from markdown code blocks first
    if content.startswith("```json"):
        lines = content.split("\n")
        content = "\n".join(lines[1:-1]) if len(lines) > 2 else content
    elif content.startswith("```"):
        lines = content.split("\n")
        content = "\n".join(lines[1:-1]) if len(lines) > 2 else content
    
    # Find the first '[' character
    start_idx = content.find('[')
    if start_idx == -1:
        return None
    
    # Try to parse JSON starting from the first '['
    # We'll try progressively longer substrings until we get valid JSON
    for end_idx in range(len(content), start_idx, -1):
        try:
            json_str = content[start_idx:end_idx].strip()
            # Remove trailing commas if present
            json_str = re.sub(r',\s*\]', ']', json_str)
            entities = json.loads(json_str)
            if isinstance(entities, list):
                return entities
        except (json.JSONDecodeError, ValueError):
            continue
    
    return None


def extract_entities_llm(text: str) -> List[dict]:
    if not text.strip():
        return []

    prompt = f"""Extract named entities from the following text. Return ONLY a JSON array, no other text.

Entity types to extract:
- ORG: Organizations, companies, institutions (e.g., "OpenAI", "Microsoft", "Stanford University")
- PERSON: People's names (e.g., "John Smith", "Einstein")
- LOCATION: Places, cities, countries (e.g., "San Francisco", "USA", "Paris")

Output format (JSON array only):
[
  {{"text": "entity name", "type": "ORG"}},
  {{"text": "entity name", "type": "PERSON"}}
]

If no entities found, return: []

Text to analyze:
{text}

JSON:"""

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "stream": False
    }

    try:
        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=300
        )
        response.raise_for_status()

        content = response.json()["message"]["content"]
        
        if not content or not content.strip():
            logger.error("LLM extraction failed: Empty response from LLM")
            return []

        # Extract JSON array from content (handles extra text after JSON)
        entities = _extract_json_array(content)
        
        if entities is None:
            logger.error(f"LLM extraction failed: Could not extract valid JSON array")
            logger.error(f"Response content: {content[:200]}")
            return []
        
        return entities

    except json.JSONDecodeError as e:
        logger.error(f"LLM extraction failed: Invalid JSON - {e}")
        logger.error(f"Response content: {content[:200] if 'content' in locals() else 'N/A'}")
        return []
    except Exception as e:
        logger.error(f"LLM extraction failed: {e}")
        return []
