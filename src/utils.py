import logging
from pdf2image import convert_from_path
import pytesseract
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np
from constants import COA_FILE_PATH, OPENAI_API_KEYS_PATH, INVOICE_PATH

# Configure logging
logging.basicConfig(level=logging.INFO, filename="audit_log.log")

# Start OpenAI client
client = OpenAI(api_key=OPENAI_API_KEYS_PATH)

# Step 1: Tesseract OCR
def extract_text_with_tesseract(file_path):
    """
    Extracts text from a PDF using Tesseract OCR only.
    """
    text_content = ""
    try:
        images = convert_from_path(file_path)
        for page_num, image in enumerate(images, start=1):
            text = pytesseract.image_to_string(image)
            text_content += f"\n--- Page {page_num} ---\n{text}"
    except Exception as e:
        logging.error(f"Tesseract OCR extraction failed: {e}")
        raise e
    return text_content

# Step 2: Parse COA
def parse_coa(file_path):
    """
    Parses the Chart of Accounts (COA) into a structured dictionary.
    """
    coa_mapping = {}
    with open(file_path, 'r') as f:
        for line in f:
            if " - " in line:
                parts = line.split(" - ")
                if len(parts) > 1:
                    account_code = parts[0].strip()
                    description = parts[1].strip()
                    keywords = description.lower().split()
                    for keyword in keywords:
                        coa_mapping[keyword] = account_code
    return coa_mapping

# Step 3: Create COA Embeddings
def coa_embeddings(coa_file_path, embedding_model='all-MiniLM-L6-v2'):
    """
    Converts the COA into embeddings for semantic search.
    """
    model = SentenceTransformer(embedding_model)
    coa_entries = []
    with open(coa_file_path, 'r') as file:
        for line in file:
            if " - " in line:
                account_code, description = line.split(" - ", 1)
                coa_entries.append({
                    "account_code": account_code.strip(),
                    "description": description.strip(),
                    "embedding": model.encode(description.strip())
                })
    return coa_entries

# Step 4: Retrieve Relevant COA Entries
def retrieve_relevant_coa(line_item_description, coa_entries, top_k=3):
    """
    Retrieves the most relevant COA entries for a given line item description.
    """
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode(line_item_description)
    scores = [
        cosine_similarity([query_embedding], [entry['embedding']])[0][0]
        for entry in coa_entries
    ]
    top_indices = np.argsort(scores)[-top_k:][::-1]
    return [coa_entries[i] for i in top_indices]

# Step 5: LLM-Based Classification
def classify_line_item_with_rag(line_item, coa_context):
    """
    Uses LLM with retrieved COA context to classify the line item.
    """
    coa_context_str = "\n".join(
        [f"{entry['account_code']}: {entry['description']}" for entry in coa_context]
    )
    
    prompt = f"""
    You are an AI assistant helping classify accounting entries according to German GAAP.
    Line Item Description: "{line_item['description']}"
    Amount: {line_item['amount']}
    Relevant Chart of Accounts Entries:
    {coa_context_str}
    
    Assign the most appropriate account code from the list above.
    Your response should only contain the account code (e.g., "5401-IT").
    """
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an expert in German GAAP accounting."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

# Step 6: Generate Journal Entries
def rag_pipeline(invoice_data, coa_file_path):
    """
    Full pipeline for processing an invoice using RAG.
    Ensures German GAAP compliance.
    """
    # Prepare COA
    coa_entries = coa_embeddings(coa_file_path)  # Generate embeddings once
    journal_entries = []
    vat_account = "1501"  # Input VAT account
    ap_account = "2000"  # Accounts Payable account
    vat_rate = 0.19  # Default VAT rate (19%)

    for line_item in invoice_data["line_items"]:
        # Retrieve relevant COA context
        coa_context = retrieve_relevant_coa(line_item["description"], coa_entries)

        # Classify line item with LLM
        account_code = classify_line_item_with_rag(line_item, coa_context)
        
        # Handle prepayments and accruals directly in the classification
        if "prepaid" in line_item["description"].lower() or "annual" in line_item["description"].lower():
            monthly_allocation = line_item["amount"] / 12
            for month in range(12):
                journal_entries.append({
                    "account_code": "1203",  # Prepaid Expenses
                    "debit": monthly_allocation,
                    "credit": 0.0,
                    "description": f"Prepayment allocation for month {month + 1}"
                })
            journal_entries.append({
                "account_code": "2000",  # Accounts Payable
                "debit": 0.0,
                "credit": line_item["amount"],
                "description": "Prepaid Expense Credit"
            })
        elif "accrued" in line_item["description"].lower() or "unpaid" in line_item["description"].lower():
            journal_entries.append({
                "account_code": "6101",  # Accrued Expenses
                "debit": line_item["amount"],
                "credit": 0.0,
                "description": "Accrued liability"
            })
        else:
            journal_entries.append({
                "account_code": account_code,
                "debit": line_item["amount"],
                "credit": 0.0,
                "description": line_item["description"]
            })

    # VAT handling
    total_vat = invoice_data.get("vat_amount", 0.0)
    if total_vat > 0:
        journal_entries.append({"account_code": vat_account, "debit": total_vat, "credit": 0.0, "description": "Input VAT"})
    
    # Accounts Payable
    total_invoice = invoice_data.get("total", 0.0)
    journal_entries.append({"account_code": ap_account, "debit": 0.0, "credit": total_invoice, "description": "Accounts Payable"})
    
    return journal_entries

