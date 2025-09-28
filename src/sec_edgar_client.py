import requests
from typing import Tuple, Optional, Dict, List

# --- Constants for SEC EDGAR APIs ---
CIK_LOOKUP_URL = 'https://www.sec.gov/files/company_tickers.json'
# Template for fetching all submission metadata for a given CIK
SUBMISSIONS_URL_TEMPLATE = 'https://data.sec.gov/submissions/CIK{}.json'
# Base URL for accessing the actual document files
ARCHIVES_BASE_URL = 'https://www.sec.gov/Archives/edgar/data/'
# Standard headers required by the SEC for programmatic access
SEC_HEADERS = {'user-agent': 'MLT AA Abdullahi.aabdii@gmail.com'}

class SecEdgarClient:
    """
    Client for retrieving CIK data and fetching SEC EDGAR filings (10-K, 10-Q).
    """
    def __init__(self):
        self.company_name: Dict[str, str] = {}   # Maps lowercased name -> CIK
        self.stock_ticker: Dict[str, str] = {}   # Maps lowercased ticker -> CIK
        # Maps padded CIK -> (cik, name, ticker)
        self.cik_data: Dict[str, Tuple[str, str, str]] = {} 
        
        # Load CIK mapping upon initialization
        self._load_cik_data()

    def _fetch_data(self, url: str) -> Dict:
        """Helper method to handle API requests with required headers and error checking."""
        print(f"Fetching data from: {url}")
        try:
            r = requests.get(url, headers=SEC_HEADERS)
            r.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            return r.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data from {url}: {e}")
            raise

    def _load_cik_data(self):
        """Loads and processes the company_tickers.json into lookup dictionaries."""
        file_json = self._fetch_data(CIK_LOOKUP_URL)
        
        # populate company_name, stock_ticker and cik_data
        for entry in file_json.values():
            cik = entry.get('cik_str')
            ticker = entry.get('ticker')
            name = entry.get('title')
            if not cik or not ticker or not name:
                continue

            # CIKs must be 10 digits with leading zeros for the submissions API
            padded_cik = str(cik).zfill(10)
            name = name.strip()
            ticker = ticker.strip()

            self.cik_data[padded_cik] = (padded_cik, name, ticker)
            self.company_name[name.lower()] = padded_cik
            self.stock_ticker[ticker.lower()] = padded_cik

    def _pad_cik(self, cik_str: str) -> str:
        """Ensures a CIK string is 10 digits, zero-padded."""
        return str(cik_str).zfill(10)

    def name_to_cik(self, name: str) -> Tuple[str, str, str]:
        """Lookup by company name (case-insensitive). Returns (cik, name, ticker)."""
        key = name.lower()
        if key not in self.company_name:
            raise KeyError(f"Company not found: '{name}'")
        cik = self.company_name[key]
        return self.cik_data[cik]

    def ticker_to_cik(self, ticker: str) -> Tuple[str, str, str]:
        """Lookup by stock ticker (case-insensitive). Returns (cik, name, ticker)."""
        key = ticker.lower()
        if key not in self.stock_ticker:
            raise KeyError(f"Ticker not found: '{ticker}'")
        cik = self.stock_ticker[key]
        return self.cik_data[cik]

    # --- New Methods for Submission History and Document Retrieval ---

    def get_submission_history(self, cik: str) -> Dict:
        """
        Fetches the full submission history JSON for a given CIK.
        This contains metadata for all filings (10-K, 10-Q, 8-K, etc.).
        """
        padded_cik = self._pad_cik(cik)
        url = SUBMISSIONS_URL_TEMPLATE.format(padded_cik)
        return self._fetch_data(url)

    def find_latest_filing_metadata(self, cik: str, form_type: str) -> Optional[Dict]:
        """
        Finds the metadata (accessionNumber, primaryDocument, etc.) for the 
        most recent filing of a specific form type (e.g., '10-Q' or '10-K').

        Args:
            cik: The CIK of the company.
            form_type: The form type to search for (e.g., '10-Q').

        Returns:
            A dictionary of the filing's metadata or None if not found.
        """
        history = self.get_submission_history(cik)

        # SEC submissions API returns filing data as parallel arrays under 'recent'
        filings_data: Dict[str, List] = history.get('filings', {}).get('recent', {})

        # Check if we have filing data and the keys we need
        if not filings_data or 'form' not in filings_data:
            print(f"Warning: No recent filing data structure found for CIK {cik}.")
            return None

        # The lists are guaranteed to be of the same length, indexed by position
        forms: List[str] = filings_data.get('form', [])

        # Search the 'form' list for the latest occurrence of the desired form_type
        # The list is sorted from newest to oldest by filing date
        for i, form in enumerate(forms):
            if form == form_type:
                # Found the latest matching filing. Assemble its metadata dictionary.
                # Accession numbers need hyphens added back for document URL construction later
                accession_no_hyphens = filings_data.get('accessionNumber', [])[i]

                return {
                    'accessionNumber': accession_no_hyphens,
                    'filingDate': filings_data.get('filingDate', [])[i],
                    'reportDate': filings_data.get('reportDate', [])[i],
                    'primaryDocument': filings_data.get('primaryDocument', [])[i],
                    # Add other keys as needed, like 'primaryDocumentDescription'
                }

        print(f"Warning: No recent filing of type '{form_type}' found for CIK {cik}.")
        return None

    def get_filing_document(self, cik: str, filing_metadata: Dict) -> Optional[str]:
        """
        Retrieves the actual text/HTML content of the primary document from a filing.

        Args:
            cik: The company's CIK.
            filing_metadata: The dictionary containing 'accessionNumber' and 'primaryDocument'.
        
        Returns:
            The raw text/HTML content of the document as a string.
        """
        accession_number = filing_metadata.get('accessionNumber')
        document_name = filing_metadata.get('primaryDocument')

        if not accession_number or not document_name:
            print("Error: Metadata missing accessionNumber or primaryDocument.")
            return None

        # 1. Strip hyphens from the accession number for the path
        accession_no_hyphens = accession_number.replace('-', '')

        # 2. Construct the full archive URL
        # Format: BASE_URL/CIK/ACCESSION_NO_HYPHENS/DOCUMENT_NAME
        url = f"{ARCHIVES_BASE_URL}{cik}/{accession_no_hyphens}/{document_name}"

        # 3. Fetch the document content (expecting text/HTML)
        # Note: We use a separate request since the content type is HTML, not JSON
        print(f"Fetching document from: {url}")
        try:
            r = requests.get(url, headers=SEC_HEADERS)
            r.raise_for_status()
            # The document content (usually raw HTML) is returned as text
            return r.text
        except requests.exceptions.RequestException as e:
            print(f"Error retrieving document from {url}: {e}")
            return None


# --- Example Usage ---

if __name__ == '__main__':
    # Initialize the client (this automatically loads all CIK data)
    sec_client = SecEdgarClient()
    print("-" * 50)
    print("--- CIK Lookup Examples ---")
    
    # 1. Lookup CIK for Apple and Microsoft
    apple_info = sec_client.ticker_to_cik('AAPL')
    msft_info = sec_client.name_to_cik('Microsoft Corp')
    print(f"Apple Info: {apple_info}")
    print(f"Microsoft Info: {msft_info}")
    print("-" * 50)
    
    # 2. Fully automate the 'Find the latest 10-Q' step for Apple
    apple_cik = apple_info[0] # Padded CIK: '0000320193'
    form_type = '10-Q'
    
    print(f"--- Finding Latest {form_type} for Apple (CIK: {apple_cik}) ---")
    
    # Step A: Get metadata for the latest 10-Q
    latest_10q_metadata = sec_client.find_latest_filing_metadata(apple_cik, form_type)
    
    if latest_10q_metadata:
        print(f"\nFound latest {form_type} filed on: {latest_10q_metadata['filingDate']}")
        print(f"Accession Number: {latest_10q_metadata['accessionNumber']}")
        
        # Step B: Retrieve the actual document content
        document_content = sec_client.get_filing_document(apple_cik, latest_10q_metadata)
        
        if document_content:
            # Print the first 500 characters of the raw HTML content
            print("\n--- Document Content Snippet (First 500 characters) ---")
            print(document_content[:500] + '...')
            print("-" * 50)
            
            # This 'document_content' variable now holds the raw 10-Q HTML 
            # and is ready for the next step: cleaning and embedding for your RAG system.
        else:
            print("Failed to retrieve document content.")
    else:
        print(f"Could not find any {form_type} filings for Apple.")
