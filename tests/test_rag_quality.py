import numpy as np
from unittest.mock import patch

from backend import parser
from backend.rag_pipeline import run_rag


def test_clean_chunks_removes_boilerplate_and_artifacts():
    sample = """NEW MESSAGE
Copyright Maven Analytics, LLC
Power Query Editor
*Copyright Maven Analytics, LLC* 
How do we shape and transform tables?
In Power Query Editor, you can promote headers, choose columns, remove rows, sort, and add custom columns.
HEY THIS IS IMPORTANT!
This is a valid transformation step.
"""

    chunks = parser.split_text(sample, chunk_size=120, overlap=20)
    cleaned = parser.clean_chunks(chunks)

    assert cleaned
    assert all('new message' not in c.lower() for c in cleaned)
    assert all('copyright maven analytics' not in c.lower() for c in cleaned)
    assert any('Power Query Editor' in c for c in cleaned)
    assert any('transform tables' in c.lower() for c in cleaned)


def test_split_text_keeps_sentences_intact_when_possible():
    sample = "Here is a first sentence. Here is a second sentence. Here is a third sentence."

    chunks = parser.split_text(sample, chunk_size=80, overlap=20)

    assert any('first sentence' in chunk.lower() and 'second sentence' in chunk.lower() for chunk in chunks)
    assert all(len(chunk.strip()) > 0 for chunk in chunks)


def test_run_rag_rejects_irrelevant_question():
    docs = [
        'Power Query Editor lets you shape and transform tables by promoting headers, choosing columns, removing rows, sorting values, and adding custom columns.',
        'Merging queries combines data from different tables using a key column, while appending queries stacks rows from similar tables.',
        'Pivoting rotates data from rows into columns, and unpivoting converts columns into rows for analysis.',
        'A territory key helps map rows to regions and is used in data filtering and reporting.',
        'Filter context controls how measures and tables behave when calculations are evaluated in Power BI.'
    ]

    mock_embeddings = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
    with patch('backend.rag_pipeline.generate_embeddings', return_value=mock_embeddings), \
         patch('backend.rag_pipeline.search', return_value=(np.array([0, 1, 2, 3, 4]), np.array([0.12, 0.10, 0.09, 0.08, 0.07]))):
        result = run_rag('What is the capital of India?', docs, top_k=5)

    assert 'couldn\'t find relevant information' in result['answer'].lower()
    assert result['citations'] == []
    assert result['chunk_ids'] == []
