# Emblematic-Accounting-Agent

The following agent takes the accounting data from one sample invoice and sort them into journal entries automatically.

## Agent Reasoning Process

This Accounting Agent follows accounting principle strictly linked to the [German CAAP](https://fairfinancialreporting.com/f/basics-of-german-gaap) applied inside the different functions of the designed scripts. 

- Parses invoices using [Tesseract](https://github.com/tesseract-ocr/tesseract) for text extraction.
- Maps invoice data to the correct Chart of Accounts (COA).
- Generates journal entries compliant with accounting principles of German GAAP using a hybrid search approach. 

## Setup

To launch the agent and sort out the entries of any invoice for accounting, you can run from your terminal the following.

```bash
curl -sSL https://install.python-poetry.org | python3 -
poetry install
export PATH="/Users/username/.local/bin:$PATH"
poetry shell
export OPENAI_API_KEY="your_openai_api_key_here"
poetry run python src/main.py
```

### Test and Evaluation

To demonstrate the system's behavior, we include the following tests for parsing, validation, mapping, and journal entries generation:

1. **Invoice Text Extraction**:
   - To ensure that the OCR engine (Tesseract) correctly extracts text from a sample invoice PDF.
2. **COA Parsing and Embedding Creation**:
   - Assess that the Chart of Accounts (COA) is correctly parsed and converted into embeddings for semantic search.
3. **Retrieval of Relevant COA Entries**:
   - To test the retrieval of the most relevant COA entries for a given line item description.
4. **LLM-Based Classification**:
   - Validates individual components like invoice validation, COA mapping, and currency conversion.
5. **RAG Pipeline Integration**:
   - To test the end-to-end behavior of the RAG pipeline, from text extraction to journal entry generation..
6. **Evaluation Framework**:
   - To validate the performance of the agent using a tailored evaluation framework.

To run the test locally, you can run the following command in your Terminal inside the current repository:

```bash
poetry run pytest test/test_agent.py
```

For each test case, detailed logs are generated, providing insights into the behavior of the system.

### Final Considerations
This use case comes in handy since there are no Zero Shot models available on Hugging Face to assess invoices and insert them in journal entry respecting the German CAAP. 
Although, a model like [FinBERT](https://huggingface.co/ProsusAI/finbert) is already trained on financial data but without being compliant specifically for the client in the scope of this repository. 
Other Document AI models that can be leveraged in some part of our workflow includes LayoutLM, Donut, or a generic OCR pipeline for extracting structured invoice data.

German GAAP (HGB) is fairly specific. A model would need domain training data with invoices, ledgers, and correct classification examples for Germany, which is not typically open-sourced and this is where the value propoposition of Emblematic comes in handy. Training such a framework like in the repo with different invoices can achieve a higher accuracy level and create a model that can be used in the German market for journal entry procedures. 
