import openai
import logging
from forex_python.converter import CurrencyRates

# Function to map invoice data to the appropriate Chart of Accounts entry
def map_to_coa(invoice_data, coa):
    """
    Maps the invoice data to the appropriate Chart of Accounts (COA) entry.
    Uses LLM for classification and COA matching.
    """
    # Example: Simple rule-based mapping for specific descriptions
    description = invoice_data["Description"].lower()

    if "software" in description:
        account = coa.get("Software Expenses", "Miscellaneous Expenses")
    elif "material" in description:
        account = coa.get("Raw Materials", "Miscellaneous Expenses")
    else:
        # Fallback to LLM-based classification
        prompt = f"""
        Invoice Details:
        Supplier: {invoice_data['Supplier']}
        Amount: {invoice_data['Amount']} {invoice_data['Currency']}
        Description: {invoice_data['Description']}
        VAT: {invoice_data['VAT']}%

        Map this invoice to the most appropriate Chart of Accounts entry.
        Chart of Accounts: {coa}
        """
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=150
        )
        account = response.choices[0].text.strip()

    return account

# Function to validate invoice data
def validate_invoice(invoice_data):
    """
    Validates the completeness and consistency of the invoice data.
    Raises an exception if any required field is missing or invalid.
    """
    required_fields = ["Supplier", "Date", "Amount", "Currency", "Description", "VAT"]
    for field in required_fields:
        if not invoice_data.get(field):
            logging.warning(f"Missing field: {field} in invoice: {invoice_data}")
            raise ValueError(f"Missing required field: {field}")

    # Validate VAT rates (German VAT: 0%, 7%, 19%)
    if invoice_data["VAT"] not in [0, 7, 19]:
        logging.warning(f"Invalid VAT rate: {invoice_data['VAT']} in invoice: {invoice_data}")
        raise ValueError(f"Invalid VAT rate: {invoice_data['VAT']}")

    logging.info("Invoice validated successfully.")
    return True

def generate_journal_entry(invoice_data, mapped_account):
    """
    Generates a journal entry compliant with German GAAP.
    """
    amount = invoice_data["Amount"]
    vat = amount * (invoice_data["VAT"] / 100)
    net_amount = amount - vat

    journal_entry = {
        "Debit": {
            mapped_account: net_amount,
            "VAT Input Tax": vat
        },
        "Credit": {
            "Accounts Payable": amount
        }
    }

    return journal_entry

def convert_currency(amount, from_currency, to_currency="EUR"):
    """
    Converts the given amount from one currency to another using real-time exchange rates.
    """
    c = CurrencyRates()
    try:
        converted_amount = c.convert(from_currency, to_currency, amount)
        logging.info(f"Converted {amount} {from_currency} to {converted_amount} {to_currency}.")
        return converted_amount
    except Exception as e:
        logging.error(f"Currency conversion failed: {e}")
        raise ValueError(f"Currency conversion failed: {e}")