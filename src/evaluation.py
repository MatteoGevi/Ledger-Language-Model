import logging
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from utils import retrieve_relevant_coa, classify_line_item_with_rag

# Configure logging for evaluation
logging.basicConfig(level=logging.INFO, filename="evaluation_log.log")

# Step 1: Evaluate COA Embedding Quality
def evaluate_coa_embeddings(coa_entries):
    """
    Evaluates the quality of COA embeddings by calculating cosine similarity between semantically similar pairs.
    Handles real-world mismatches using approximate matching and debugging outputs.
    """
    similar_pairs = [
        ("Prepaid Expenses", "Prepaid Software Subscriptions"),
        ("Software Subscriptions (IT)", "IT Support and Maintenance (IT)"),
    ]
    scores = []

    print("Starting evaluation of COA embeddings...")
    for pair in similar_pairs:
        # Find exact or approximate matches for the descriptions
        embedding_1 = next(
            entry['embedding'] for entry in coa_entries if pair[0].lower() in entry['description'].lower()
        )
        embedding_2 = next(
            entry['embedding'] for entry in coa_entries if pair[1].lower() in entry['description'].lower()
        )
        
        # Compute cosine similarity
        score = cosine_similarity([embedding_1], [embedding_2])[0][0]
        scores.append(score)
        print(f"Similarity between '{pair[0]}' and '{pair[1]}': {score:.4f}")

    average_similarity = np.mean(scores) if scores else 0
    print(f"\nAverage Cosine Similarity: {average_similarity:.4f}")
    print(f"Scores: {scores}")

    return {"average_cosine_similarity": average_similarity, "scores": scores}

# Step 2: Evaluate LLM Classification Accuracy
def evaluate_llm_classification(line_items, expected_entries, coa_entries, classify_line_item_with_rag):
    correct = 0
    total_items = len(line_items)

    for item, expected in zip(line_items, expected_entries):
        retrieved_coa = retrieve_relevant_coa(item["description"], coa_entries, top_k=3)
        predicted_code = classify_line_item_with_rag(item, retrieved_coa)
        if predicted_code == expected["account_code"]:
            correct += 1

    accuracy = correct / total_items if total_items > 0 else 0.0
    return {"accuracy": accuracy}

# Step 3: Validate Journal Entries
def validate_journal_entries(
    journal_entries, 
    total_invoice,  # or call it net_invoice if that’s what you’re actually passing
    vat_rate=0.19
):
    total_debit = sum(e.get("debit", 0.0) for e in journal_entries)
    total_credit = sum(e.get("credit", 0.0) for e in journal_entries)
    is_balanced = round(total_debit, 2) == round(total_credit, 2)

    # Example check for VAT lines
    vat_entries = [e for e in journal_entries if "Input VAT" in e["description"]]
    vat_compliant = all(
        abs(e["debit"] - (total_invoice * vat_rate)) < 0.01 for e in vat_entries
    )

    return {
        "is_balanced": is_balanced,
        "vat_compliant": vat_compliant,
        "total_debit": total_debit,
        "total_credit": total_credit
    }


# Step 4: Evaluate End-to-End Pipeline Accuracy
def evaluate_pipeline_accuracy(generated_entries, expected_entries):
    """
    Compare generated journal entries with expected outputs for accuracy.
    """
    correct = 0
    for generated, expected in zip(generated_entries, expected_entries):
        if (
            generated["account_code"] == expected["account_code"] and
            round(generated["debit"], 2) == round(expected["debit"], 2) and
            round(generated["credit"], 2) == round(expected["credit"], 2)
        ):
            correct += 1
    accuracy = correct / len(expected_entries)
    return {"pipeline_accuracy": accuracy}

# Comprehensive Evaluation Framework
def evaluate_agent(coa_entries, test_data, expected_entries, pipeline_output):
    """
    Comprehensive evaluation framework for the RAG pipeline.
    Expects:
      - coa_entries: list of COA dict with 'embedding'
      - test_data: dict with 'line_items', 'vat_amount', 'total'
      - expected_entries: list of ground truth dicts
      - pipeline_output: list of generated journal entries
    """
    results = {}

    # 1. Evaluate COA Embedding Quality
    results["coa_embedding"] = evaluate_coa_embeddings(coa_entries)

    # 2. Evaluate LLM Classification (just the line items, not VAT/AP lines)
    results["llm_classification"] = evaluate_llm_classification(
        line_items=test_data["line_items"],
        expected_entries=expected_entries[:len(test_data["line_items"])],  # just the first 3 if you like
        coa_entries=coa_entries,
        classify_line_item_with_rag=classify_line_item_with_rag
    )

    # 3. Validate Journal Entries (balance check, VAT check)
    results["journal_validation"] = validate_journal_entries(
        journal_entries=pipeline_output,
        total_invoice=test_data["total"]
    )

    # 4. Compare final pipeline output to the entire expected entries
    results["pipeline_accuracy"] = evaluate_pipeline_accuracy(
        generated_entries=pipeline_output,
        expected_entries=expected_entries
    )

    return results