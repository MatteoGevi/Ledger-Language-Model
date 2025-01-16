from main import process_invoice

def test_process_invoice():
    coa = {"Raw Materials": "Account 3001", "Software Expenses": "Account 5001"}
    invoice_data = {
        "Supplier": "Eco Supplies",
        "Date": "2024-12-01",
        "Amount": 1200,
        "Currency": "USD",
        "Description": "Raw materials for manufacturing",
        "VAT": 19
    }

    journal_entry = process_invoice(invoice_data, coa)
    assert journal_entry is not None
    assert "Debit" in journal_entry
    assert "Credit" in journal_entry
    assert journal_entry["Debit"]["Account 3001"] == 1008.4  # Net amount in EUR