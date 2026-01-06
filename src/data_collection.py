import yfinance as yf

def fetch_crypto_data():
    data = yf.download("BTC-USD", start="2020-01-01", end="2025-01-01")
    data.to_csv("data/btc_data.csv")
    print("Data downloaded successfully")

if __name__ == "__main__":
    fetch_crypto_data()
