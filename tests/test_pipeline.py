import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from models import Document
from pipeline import run_pipeline



def test_pipeline_empty():
    doc = Document(id="2", text=" ")
    result = run_pipeline(doc)
    assert result.entities == []

def test_pipeline_llm_entities():
    doc = Document(
        id="1",
        text="OpenAI builds AI models in San Francisco."
    )
    result = run_pipeline(doc)

    texts = [e["text"] for e in result.entities]

    assert "OpenAI" in texts
    assert result.confidence >= 0.7