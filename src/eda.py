import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import mplcursors
from preprocessing import preprocess_data

# -------------------------------------------------
# LOAD & PREPARE DATA
# -------------------------------------------------

df = preprocess_data()

# Convert Date back to datetime for plotting & hover
df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y")

# -------------------------------------------------
# CREATE PLOT
# -------------------------------------------------

fig, ax = plt.subplots(figsize=(14, 6))

line, = ax.plot(
    df["Date"],
    df["Close"],
    linewidth=2,
    color="blue"
)

# -------------------------------------------------
# X-AXIS: YEAR → JULY → YEAR FORMAT
# -------------------------------------------------

# Major ticks: Year
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

# Minor ticks: July
ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=7))
ax.xaxis.set_minor_formatter(mdates.DateFormatter("%b"))

# -------------------------------------------------
# Y-AXIS: ACTUAL PRICE (NO EXPONENTIAL)
# -------------------------------------------------

ax.yaxis.set_major_formatter(
    FuncFormatter(lambda x, _: f"{int(x):,}")
)

# -------------------------------------------------
# LABELS & STYLING
# -------------------------------------------------

ax.set_title("Bitcoin Price Trend", fontsize=14)
ax.set_xlabel("Year / Month", fontsize=11)
ax.set_ylabel("Price (USD)", fontsize=11)

ax.grid(True, which="both", linestyle="--", alpha=0.4)

plt.tight_layout()

# -------------------------------------------------
# HOVER FUNCTIONALITY (FIXED 1970 BUG)
# -------------------------------------------------

cursor = mplcursors.cursor(line, hover=True)

@cursor.connect("add")
def on_hover(sel):
    # Convert matplotlib float date → real datetime
    date = mdates.num2date(sel.target[0])
    price = sel.target[1]

    sel.annotation.set_text(
        f"Date: {date.strftime('%d %B %Y')}\n"
        f"Price: ${int(price):,}"
    )

    sel.annotation.get_bbox_patch().set(
        fc="white",
        alpha=0.9
    )

# -------------------------------------------------
# SHOW PLOT
# -------------------------------------------------

plt.show()
