# Emblematic-Accounting-Agent

The following agent takes the accounting data from one sample invoice and sort them into journal entries automatically. 

## Agent Reasoning Process

Our Agent follows accounting principle strictly linked to the 

- Parses invoices using OCR or text extraction.
- Maps invoice data to the correct Chart of Accounts (COA).
- Generates journal entries compliant with accounting principles such as German GAAP.
- Evaluates performance using historical journal data.

## Setup

To launch the agent and sort out the entries of any invoice for accounting, you can run from your terminal the following.

Once you found the document that you want the agent to parse, you just need to paste the file name as next command and the agent will sort them as journal entry and give you the output directly in the terminal. 

### Test and Evaluation

To demonstrate the system's behavior, we include the following tests:

1. **Unit Tests**:
   - Validates individual components like invoice validation, COA mapping, and currency conversion.
2. **Integration Tests**:
   - Processes an invoice from parsing to journal entry generation.
3. **Real-World Edge Cases**:
   - Handles missing fields, invalid VAT rates, and multi-currency scenarios.

To run the test locally, you can run the following command in your Terminal inside the current repository:

```bash
poetry run pytest
```

# Further Improvements

The agent works simply for one single client and it can be launched from a terminal. Both operational, business and infrastructural improvements can be advanced:

- Since the client is a SME from Munich, the use case was quite straightforward. To implement 
- To optimize the computational cost, it can be convenient to fine a Small Language Model (SLM) 


