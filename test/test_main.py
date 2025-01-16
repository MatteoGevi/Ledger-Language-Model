import pytest
from utils import validate_invoice, map_to_coa, convert_currency, generate_journal_entry

def test_validate_invoice():
    invoice_data = {
        "Supplier": "Eco Supplies",
        "Date": "2024-12-01",
        "Amount": 1000,
        "Currency": "EUR",
        "Description": "Raw materials",
        "VAT": 19
    }
    assert validate_invoice(invoice_data) is None  # Should pass without raising exceptions

def test_map_to_coa():
    coa = {"Raw Materials": "Account 3001"}
    invoice_data = {"Description": "Raw materials"}
    account = map_to_coa(invoice_data, coa)
    assert account == "Account 3001"

def test_convert_currency():
    amount = convert_currency(100, "USD", "EUR")
    assert amount > 0  # Ensure conversion returns a valid number

def test_generate_journal_entry():
    invoice_data = {"Amount": 1190, "VAT": 19}
    journal_entry = generate_journal_entry(invoice_data, "Raw Materials")
    assert "Debit" in journal_entry and "Credit" in journal_entry