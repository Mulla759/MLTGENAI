import requests

class secEdgar:
    # SEC EDGAR class retrieves and parses company data
    def __init__(self, fileurl):
        self.fileurl = fileurl
        self.company_name = {}   
        self.stock_ticker = {}   
        self.cik_data = {}       
        
        headers = {'user-agent': 'MLT AA Abdullahi.aabdii@gmail.com'}
        r = requests.get(self.fileurl, headers=headers)
        r.raise_for_status()
        self.filejson = r.json()
        self.cik_json_to_dict()

    def cik_json_to_dict(self):
        # populate company_name, stock_ticker and cik_data
        for entry in self.filejson.values():
            cik = entry.get('cik_str')
            ticker = entry.get('ticker')
            name = entry.get('title')
            if not cik or not ticker or not name:
                continue

            cik = str(cik).zfill(10)
            name = name.strip()
            ticker = ticker.strip()

            self.cik_data[cik] = (cik, name, ticker)
            self.company_name[name.lower()] = cik
            self.stock_ticker[ticker.lower()] = cik

    def name_to_cik(self, name):
        # lookup by company name (case‑insensitive)
        key = name.lower()
        if key not in self.company_name:
            raise KeyError(f"Company not found: '{name}'")
        cik = self.company_name[key]
        return self.cik_data[cik]

    def ticker_to_cik(self, ticker):
        # lookup by stock ticker (case‑insensitive)
        key = ticker.lower()
        if key not in self.stock_ticker:
            raise KeyError(f"Ticker not found: '{ticker}'")
        cik = self.stock_ticker[key]
        return self.cik_data[cik]



sec = secEdgar('https://www.sec.gov/files/company_tickers.json')

# lookups by name
print(sec.name_to_cik('Apple Inc.'))
print(sec.name_to_cik('Microsoft Corp'))
print(sec.name_to_cik('Tesla, Inc.'))


print(sec.ticker_to_cik('AAPL'))
print(sec.ticker_to_cik('MSFT'))
print(sec.ticker_to_cik('TSLA'))
print(sec.ticker_to_cik('GOOGL'))
print(sec.ticker_to_cik('AMZN'))
print(sec.ticker_to_cik('IBM'))
