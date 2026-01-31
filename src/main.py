import os
from loguru import logger
from models import Document
from pipeline import run_pipeline


os.makedirs("../logs", exist_ok=True)
logger.add("../logs/app.log", level="INFO")

if __name__ == "__main__":
    logger.info("Pipeline started")

    doc = Document(id="1", text="OpenAI builds AI models in San Francisco.")
    result = run_pipeline(doc)

    logger.info(f"Pipeline finished. Result: {result}")
