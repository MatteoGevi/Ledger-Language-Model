import os

# Get the directory where this file is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define paths to the files relative to the `src` directory
COA_FILE_PATH = os.path.join(BASE_DIR, "../data/Chart_of_Accounts(COA).txt")
INVOICE_PATH = os.path.join(BASE_DIR, "../data/journal_entries_sample.pdf")
OPENAI_API_KEYS_PATH = os.getenv("OPENAI_API_KEY")