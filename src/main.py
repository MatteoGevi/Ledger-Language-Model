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
        # Step 1: Create COA embeddings
        coa_entries = coa_embeddings(COA_FILE_PATH)

        # Step 2: OCR invoice if needed (optional)
        ocr_text = extract_text_with_tesseract(INVOICE_PATH)

        # Step 3: Define consistent invoice data
        invoice_data = {
            "line_items": [
                {"description": "annual software subscription", "amount": 1200.0},
                {"description": "office rent", "amount": 2500.0},
                {"description": "utilities payment", "amount": 300.0},
            ],
            "vat_amount": 760.0,  # 19% of 4,000
            "total": 4760.0       # net 4,000 + VAT 760
        }

        # Generate journal entries
        pipeline_output = rag_pipeline(invoice_data, COA_FILE_PATH)
        print("Generated Journal Entries:")
        for entry in pipeline_output:
            print(entry)

        # Step 4: Define matching expected entries
        expected_entries = [
            {"account_code": "5401-IT", "debit": 1200.0, "credit": 0.0},
            {"account_code": "5201-HQ", "debit": 2500.0, "credit": 0.0},
            {"account_code": "5202-HQ", "debit": 300.0,  "credit": 0.0},
            {"account_code": "1501-EUR", "debit": 760.0,  "credit": 0.0},  # VAT
            {"account_code": "2000",    "debit": 0.0,     "credit": 4760.0}, # A/P
        ]

        # Step 5: Evaluate
        evaluation_results = evaluate_agent(
            coa_entries=coa_entries,
            test_data=invoice_data,
            expected_entries=expected_entries,
            pipeline_output=pipeline_output
        )

        print("\nEvaluation Results:")
        for key, value in evaluation_results.items():
            print(f"{key}: {value}")

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
