from typing import List

def calculate_confidence(entities: List[dict]) -> float:
    """
    Calculate confidence score based on extracted entities.
    Returns a float between 0.0 and 1.0.
    """
    if not entities:
        return 0.0
    
    # Simple confidence calculation: more entities = higher confidence
    # Base confidence of 0.7, increases with more entities (capped at 0.95)
    base_confidence = 0.7
    entity_bonus = min(len(entities) * 0.05, 0.25)
    return min(base_confidence + entity_bonus, 0.95)
