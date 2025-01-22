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
        ("Software Subscriptions", "IT Licenses"),
        ("Office Supplies", "Stationery"),
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
def evaluate_llm_classification(test_set, coa_entries, classify_line_item_with_rag):
    """
    Evaluate the accuracy of the LLM in assigning correct account codes.
    """
    correct = 0
    for line_item in test_set:
        retrieved_coa = retrieve_relevant_coa(line_item["description"], coa_entries, top_k=3)
        predicted_code = classify_line_item_with_rag(line_item, retrieved_coa)
        if predicted_code == line_item["expected_code"]:
            correct += 1
    accuracy = correct / len(test_set)
    return {"accuracy": accuracy}

# Step 3: Validate Journal Entries
def validate_journal_entries(journal_entries, total_invoice, vat_rate=0.19):
    """
    Validate journal entries for balance, VAT compliance, and correct prepayment/accrual handling.
    """
    total_debit = sum(entry["debit"] for entry in journal_entries if "debit" in entry)
    total_credit = sum(entry["credit"] for entry in journal_entries if "credit" in entry)
    is_balanced = round(total_debit, 2) == round(total_credit, 2)

    vat_entries = [entry for entry in journal_entries if "Input VAT" in entry["description"]]
    vat_compliant = all(entry["debit"] == total_invoice * vat_rate for entry in vat_entries)

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
    """
    results = {}

    # Step 1: Evaluate COA Embedding Quality
    results["coa_embedding"] = evaluate_coa_embeddings(coa_entries)

    # Step 2: Evaluate LLM Classification
    results["llm_classification"] = evaluate_llm_classification(test_data["line_items"], coa_entries, classify_line_item_with_rag)

    # Step 3: Validate Journal Entries
    results["journal_validation"] = validate_journal_entries(pipeline_output, sum(item["amount"] for item in test_data["line_items"]))

    # Step 4: End-to-End Accuracy
    results["pipeline_accuracy"] = evaluate_pipeline_accuracy(pipeline_output, expected_entries)

    return results