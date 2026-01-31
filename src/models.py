from pydantic import BaseModel
from typing import List, Dict, Any

class Document(BaseModel):
    id: str
    text: str

class ExtractionResult(BaseModel):
    entities: List[Dict[str, Any]]
    confidence: float