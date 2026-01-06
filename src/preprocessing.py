# import pandas as pd

# def preprocess_data():
#     # Read raw CSV
#     df = pd.read_csv("data/btc_data.csv")

#     # 1️⃣ Remove first two invalid rows (Ticker + Date rows)
#     df = df.iloc[2:].reset_index(drop=True)

#     # 2️⃣ Rename columns correctly
#     df.columns = ["Date", "Close", "High", "Low", "Open", "Volume"]

#     # 3️⃣ Parse Indian date format (DO NOT SORT or MODIFY ORDER)
#     df["Date"] = pd.to_datetime(
#         df["Date"],
#         format="%d-%m-%Y",
#         errors="coerce"
#     )

#     # 4️⃣ Convert numeric columns
#     for col in ["Open", "High", "Low", "Close", "Volume"]:
#         df[col] = pd.to_numeric(df[col], errors="coerce")

#     # 5️⃣ Drop only truly invalid rows
#     df = df.dropna()

#     # 6️⃣ Convert back to Indian format (final output)
#     df["Date"] = df["Date"].dt.strftime("%d-%m-%Y")

#     return df


# if __name__ == "__main__":
#     df = preprocess_data()
#     print(df.head())
#     print("\nSample mapping check:")
#     print("01-01-2020 →", df.loc[df["Date"] == "01-01-2020", "Close"].values[0])
#     print("02-01-2020 →", df.loc[df["Date"] == "02-01-2020", "Close"].values[0])
import pandas as pd
import os
import yfinance as yf

def preprocess_data():
    csv_path = "data/btc_data.csv"

    # -------------------------------------------------
    # 1️⃣ Use CSV if it exists (local + cloud)
    # -------------------------------------------------
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)

        # Remove extra header rows if present
        if not df.empty and "Date" not in df.columns:
            df = df.iloc[2:].reset_index(drop=True)
            df.columns = ["Date", "Close", "High", "Low", "Open", "Volume"]

        df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y", errors="coerce")

        for col in ["Open", "High", "Low", "Close", "Volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna()
        return df

    # -------------------------------------------------
    # 2️⃣ Fallback to Yahoo Finance (safety)
    # -------------------------------------------------
    df = yf.download("BTC-USD", start="2020-01-01", progress=False)
    df.reset_index(inplace=True)
    return df
