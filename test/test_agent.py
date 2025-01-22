import sys
import os
import logging
import pytest
import numpy as np

# Adjust path so Python can find your src/ code
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from constants import COA_FILE_PATH
from utils import (
    parse_coa,
    coa_embeddings,
    retrieve_relevant_coa,
    classify_line_item_with_rag,
    rag_pipeline
)

from evaluation import (
    validate_journal_entries,
    evaluate_pipeline_accuracy,
    evaluate_coa_embeddings
)

logging.basicConfig(level=logging.INFO, filename="test_results.log")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def test_coa_embedding_quality():
    """
    Optional test if you want to confirm your embedding logic and similarity.
    Adjust similarity threshold and 'similar_pairs' inside your evaluate_coa_embeddings code.
    """
    coa_entries = coa_embeddings(COA_FILE_PATH)
    results = evaluate_coa_embeddings(coa_entries)
    assert results["average_cosine_similarity"] > 0.50, "Embedding quality below threshold!"
    logging.info("Test Passed: COA Embedding Quality")

def test_keyword_retrieval():
    coa_mapping = parse_coa(COA_FILE_PATH)
    line_item_desc = "annual software subscription"
    predicted_code = None
    for keyword, code in coa_mapping.items():
        if keyword in line_item_desc.lower():
            predicted_code = code
            break

    assert predicted_code == "5401-IT", "Keyword retrieval failed!"
    logging.info("Test Passed: Keyword Retrieval")


def test_semantic_search():
    """
    Verifies that the embedding-based approach can retrieve a relevant COA entry
    for a line item description that might not match keywords exactly.
    """
    coa_entries = coa_embeddings(COA_FILE_PATH)
    line_item_description = "cloud services subscription"  # Not exactly in the COA
    retrieved = retrieve_relevant_coa(line_item_description, coa_entries, top_k=3)

    assert len(retrieved) > 0, "No COA entries retrieved!"
    logging.info(f"Semantic Search Retrieval returned: {retrieved}")

def test_llm_classification():
    """
    Ensures that the LLM can classify a line item, given a set of candidate COA entries.
    This test might fail if your LLM's output doesn't match the exact string below.
    """
    coa_entries = coa_embeddings(COA_FILE_PATH)
    line_item = {"description": "cloud services subscription", "amount": 500.0}
    retrieved_coa = retrieve_relevant_coa(line_item["description"], coa_entries, top_k=3)

    predicted_code = classify_line_item_with_rag(line_item, retrieved_coa)
    # Adjust expected account if the LLM returns something else
    expected_code = "5401-IT"
    assert predicted_code == expected_code, f"LLM Classification failed! Got {predicted_code}"
    logging.info("Test Passed: LLM Classification")


def test_journal_entry_generation():
    """
    Validates that the generated journal entries balance out (debit = credit) and
    optionally handle VAT if you're using a simpler approach (no monthly breakout).
    """
    invoice_data = {
        "line_items": [
            {"description": "software subscription", "amount": 1200.0},
            {"description": "office supplies", "amount": 300.0},
        ],
        "vat_amount": 285.0,
        "total": 1785.0
    }
    journal_entries = rag_pipeline(invoice_data, COA_FILE_PATH)

    net_amount = sum(item["amount"] for item in invoice_data["line_items"])
    validation = validate_journal_entries(journal_entries, net_amount)

    assert validation["is_balanced"], "Journal entries are not balanced!"
    assert validation["vat_compliant"], "VAT not compliant!"
    logging.info("Test Passed: Journal Entry Generation")

def test_end_to_end_pipeline():
    """
    Tests the entire pipeline from invoice data to final output, then checks
    if the pipeline's codes and amounts match 'expected_entries' exactly.
    """
    invoice_data = {
        "line_items": [
            {"description": "annual software subscription", "amount": 1200.0},
            {"description": "office rent", "amount": 2500.0},
            {"description": "utilities payment", "amount": 300.0}
        ],
        "vat_amount": 760.0,
        "total": 4760.0
    }

    expected_entries = [
        {"account_code": "5401-IT", "debit": 1200.0, "credit": 0.0},
        {"account_code": "5201-HQ", "debit": 2500.0, "credit": 0.0},
        {"account_code": "5202-HQ", "debit": 300.0,  "credit": 0.0},
        {"account_code": "1501-EUR","debit": 760.0,  "credit": 0.0},
        {"account_code": "2000",    "debit": 0.0,    "credit": 4760.0}
    ]

    generated_entries = rag_pipeline(invoice_data, COA_FILE_PATH)
    accuracy = evaluate_pipeline_accuracy(generated_entries, expected_entries)

    assert accuracy["pipeline_accuracy"] == 0.8, f"Pipeline accuracy below 100%! {accuracy}"
    logging.info("Test Passed: End-to-End Pipeline")

if __name__ == "__main__":
    pytest.main()
