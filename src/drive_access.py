from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2 import service_account
import io
import os

SCOPES = ["https://www.googleapis.com/auth/drive"]
CREDENTIALS_FILE = "path/to/credentials.json"

def get_drive_service():
    """
    Authenticate and return the Google Drive API service object.
    """
    credentials = service_account.Credentials.from_service_account_file(
        CREDENTIALS_FILE, scopes=SCOPES
    )
    service = build("drive", "v3", credentials=credentials)
    return service

def download_file(file_id, file_name, output_folder="data/"):
    """
    Download a file from Google Drive using its file ID.
    """
    service = get_drive_service()
    request = service.files().get_media(fileId=file_id)
    file_path = os.path.join(output_folder, file_name)

    os.makedirs(output_folder, exist_ok=True)
    with io.FileIO(file_path, "wb") as fh:
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
            print(f"Download {int(status.progress() * 100)}%.")
    print(f"File downloaded to {file_path}.")
    return file_path

def list_files_in_drive(folder_id=None):
    """
    List files in a Google Drive folder or root directory.
    """
    service = get_drive_service()
    query = f"'{folder_id}' in parents" if folder_id else None
    results = service.files().list(q=query, fields="files(id, name)").execute()
    files = results.get("files", [])
    for file in files:
        print(f"Name: {file['name']}, ID: {file['id']}")
    return files

if __name__ == "__main__":
    # Example: List files in Google Drive
    print("Listing files in Google Drive...")
    files = list_files_in_drive()

    # Example: Download a specific file
    file_id = "YOUR_FILE_ID"
    file_name = "sample_invoice.pdf"
    download_file(file_id, file_name)