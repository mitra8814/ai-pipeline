from loguru import logger
from models import Document, ExtractionResult
from extractors.llm import extract_entities_llm
from scoring.confidence import calculate_confidence

def run_pipeline(doc: Document) -> ExtractionResult:
    logger.info(f"Running pipeline for doc id: {doc.id}")

    entities = extract_entities_llm(doc.text)
    confidence = calculate_confidence(entities)

    return ExtractionResult(
        entities=entities,
        confidence=confidence
    )