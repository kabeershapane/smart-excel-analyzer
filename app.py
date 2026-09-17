
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Smart Excel Analyzer", page_icon="📊", layout="wide")

st.title("📊 Smart Excel Analyzer")
st.caption("Upload an Excel or CSV file and get an automatic financial/data analysis.")

@st.cache_data
def load_file(uploaded_file):
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    return pd.read_excel(uploaded_file)

def find_col(columns, keywords):
    for c in columns:
        s = str(c).lower().replace("_", " ").strip()
        if any(k in s for k in keywords):
            return c
    return None

uploaded = st.file_uploader("Upload Excel (.xlsx/.xls) or CSV", type=["xlsx", "xls", "csv"])

if not uploaded:
    st.info("Upload a file to start. The app automatically looks for date, amount, debit/credit, category, and description columns.")
    st.markdown("""
    **Included analysis**
    - Dataset overview and quality checks
    - Total income/credits and expenses/debits
    - Net cash flow
    - Spending by category
    - Daily/monthly trends
    - Largest transactions
    - Simple outlier detection
    - Downloadable cleaned data
    """)
    st.stop()

try:
    df = load_file(uploaded)
except Exception as e:
    st.error(f"Could not read the file: {e}")
    st.stop()

if df.empty:
    st.warning("The uploaded file is empty.")
    st.stop()

st.subheader("1. Data preview")
st.write(f"**{len(df):,} rows × {len(df.columns):,} columns**")
st.dataframe(df.head(100), use_container_width=True)

columns = list(df.columns)

# Column detection
date_col = find_col(columns, ["date", "datetime", "timestamp", "transaction date"])
amount_col = find_col(columns, ["amount", "value", "debit", "credit", "transaction amount"])
desc_col = find_col(columns, ["description", "particular", "narration", "merchant", "details", "remarks"])
category_col = find_col(columns, ["category", "type", "expense category", "group"])

st.sidebar.header("Column mapping")
date_col = st.sidebar.selectbox("Date column", ["— None —"] + columns, index=(columns.index(date_col)+1 if date_col in columns else 0))
amount_col = st.sidebar.selectbox("Amount column", ["— None —"] + columns, index=(columns.index(amount_col)+1 if amount_col in columns else 0))
desc_col = st.sidebar.selectbox("Description / merchant column", ["— None —"] + columns, index=(columns.index(desc_col)+1 if desc_col in columns else 0))
category_col = st.sidebar.selectbox("Category column", ["— None —"] + columns, index=(columns.index(category_col)+1 if category_col in columns else 0))

work = df.copy()

# Parse date
if date_col != "— None —":
    work["_date"] = pd.to_datetime(work[date_col], errors="coerce", dayfirst=True)
else:
    work["_date"] = pd.NaT

# Parse amount
if amount_col != "— None —":
    work["_amount"] = pd.to_numeric(
        work[amount_col].astype(str).str.replace(",", "", regex=False).str.replace("₹", "", regex=False).str.strip(),
        errors="coerce"
    )
else:
    work["_amount"] = np.nan

st.subheader("2. Automatic analysis")

if work["_amount"].notna().any():
    vals = work["_amount"].dropna()
    # If separate debit/credit columns are present, the user can still map amount to one numeric field.
    # For mixed signed transaction data, negative values are expenses and positive values are income.
    income = vals[vals > 0].sum()
    expense = -vals[vals < 0].sum()
    net = vals.sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Income / Credits", f"₹{income:,.2f}")
    c2.metric("Expenses / Debits", f"₹{expense:,.2f}")
    c3.metric("Net cash flow", f"₹{net:,.2f}")
    c4.metric("Transactions", f"{len(vals):,}")

    # If amounts are all positive, offer a neutral note rather than assuming they're expenses.
    if (vals >= 0).all():
        st.warning("All detected amounts are positive. The app cannot determine income vs expense without a signed amount column or separate debit/credit information.")

    # Trend
    if work["_date"].notna().any():
        trend = work.dropna(subset=["_date", "_amount"]).copy()
        trend["_day"] = trend["_date"].dt.date
        daily = trend.groupby("_day")["_amount"].sum()

        st.markdown("### Daily net movement")
        fig, ax = plt.subplots(figsize=(10, 4))
        daily.plot(kind="bar", ax=ax)
        ax.axhline(0, linewidth=1)
        ax.set_xlabel("Date")
        ax.set_ylabel("Net movement")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        st.pyplot(fig, clear_figure=True)

    # Categories
    if category_col != "— None —":
        cat = work.dropna(subset=["_amount"]).copy()
        cat["_category"] = cat[category_col].fillna("Uncategorized").astype(str)
        expense_cat = cat[cat["_amount"] < 0].copy()
        if not expense_cat.empty:
            by_cat = expense_cat.assign(_expense=-expense_cat["_amount"]).groupby("_category")["_expense"].sum().sort_values(ascending=False).head(15)
            st.markdown("### Expenses by category")
            fig, ax = plt.subplots(figsize=(10, 5))
            by_cat.sort_values().plot(kind="barh", ax=ax)
            ax.set_xlabel("Expense")
            ax.set_ylabel("")
            plt.tight_layout()
            st.pyplot(fig, clear_figure=True)
            st.dataframe(by_cat.rename("Total expense").to_frame(), use_container_width=True)

    # Largest expenses
    expenses = work[work["_amount"] < 0].copy()
    if not expenses.empty:
        st.markdown("### Largest expenses")
        cols = [c for c in [date_col, desc_col, category_col, amount_col] if c != "— None —" and c in work.columns]
        largest = expenses.sort_values("_amount").head(15)
        st.dataframe(largest[cols] if cols else largest, use_container_width=True)

    # Outliers using IQR on expense magnitude
    if len(expenses) >= 5:
        x = -expenses["_amount"]
        q1, q3 = x.quantile([0.25, 0.75])
        iqr = q3 - q1
        threshold = q3 + 1.5 * iqr
        outliers = expenses[x >= threshold].copy()
        if not outliers.empty:
            st.markdown("### ⚠️ Unusually large expenses")
            st.caption(f"Flagged using an IQR-based rule; threshold ≈ {threshold:,.2f}. This is a statistical flag, not proof of fraud.")
            cols = [c for c in [date_col, desc_col, category_col, amount_col] if c != "— None —" and c in work.columns]
            st.dataframe(outliers[cols] if cols else outliers, use_container_width=True)
else:
    st.error("Please select a numeric Amount column in the sidebar.")

st.subheader("3. Data quality")
quality = pd.DataFrame({
    "Column": df.columns,
    "Missing values": [df[c].isna().sum() for c in df.columns],
    "Unique values": [df[c].nunique(dropna=True) for c in df.columns],
    "Data type": [str(df[c].dtype) for c in df.columns],
})
st.dataframe(quality, use_container_width=True)

st.subheader("4. Download")
csv = df.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download original data as CSV", csv, "analyzed_data.csv", "text/csv")
