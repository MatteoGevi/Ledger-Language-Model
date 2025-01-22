import logging
import os
from utils import (
    extract_text_with_tesseract,
    parse_coa,
    coa_embeddings,
    rag_pipeline
)
from evaluation import evaluate_agent
from constants import COA_FILE_PATH, INVOICE_PATH

# Configure logging for main execution
logging.basicConfig(level=logging.INFO, filename="main_log.log", format="%(asctime)s - %(levelname)s - %(message)s")
# To avoid tokenization warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main():
    try:
        # Step 1: Prepare Chart of Accounts
        logging.info("Step 1: Creating COA embeddings...")
        coa_entries = coa_embeddings(COA_FILE_PATH)
        logging.info(f"COA embeddings created successfully: {len(coa_entries)} entries.")

        # Step 2: Process Invoice
        logging.info("Step 2: Processing invoice using Tesseract OCR...")
        ocr_text = extract_text_with_tesseract(INVOICE_PATH)
        logging.info("OCR text extraction completed.")

        # Example invoice data
        invoice_data = {
            "line_items": [
                {"description": "annual software subscription", "amount": 1200.0},
                {"description": "office rent", "amount": 2500.0},
                {"description": "utilities payment", "amount": 300.0},
            ],
            "vat_amount": 285.0,
            "total": 1785.0,
        }
        logging.info(f"Invoice data structured: {invoice_data}")

        # Step 3: Run RAG Pipeline
        logging.info("Step 3: Running RAG pipeline to generate journal entries...")
        pipeline_output = rag_pipeline(invoice_data, COA_FILE_PATH)
        logging.info("Journal entries generated:")
        for entry in pipeline_output:
            logging.info(entry)
        
        # Print journal entries for visibility
        print("Generated Journal Entries:")
        for entry in pipeline_output:
            print(entry)

        # Step 4: Evaluate Results
        logging.info("Step 4: Evaluating pipeline performance...")

        # Prepare test data and expected results for evaluation
        expected_entries = [
            {"description": "annual software subscription", "amount": 1200.0, "account_code": "5401"},
            {"description": "office rent", "amount": 2500.0, "account_code": "5201"},
            {"description": "utilities payment", "amount": 300.0, "account_code": "5202"},
        ]

        # Run evaluation framework
        evaluation_results = evaluate_agent(
            coa_entries=coa_entries,
            test_data=invoice_data,
            expected_entries=expected_entries,
            pipeline_output=pipeline_output,
        )
        logging.info("Evaluation results:")
        for key, value in evaluation_results.items():
            logging.info(f"{key}: {value}")

        # Print evaluation results for user visibility
        print("\nEvaluation Results:")
        for key, value in evaluation_results.items():
            print(f"{key}: {value}")

    except Exception as e:
        logging.error(f"An error occurred during the main execution: {e}")
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
