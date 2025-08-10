# --- Optional safety: adjust rich if Cloud ever downgrades Streamlit later ---
try:
    import streamlit as st, rich
    from packaging.version import Version
    # Streamlit < 1.42 requires rich < 14; 1.42+ doesn't require rich.
    # If a future image downgrades Streamlit, keep the combo compatible:
    import sys, subprocess
    if Version(st.__version__) < Version("1.42.0") and Version(rich.__version__) >= Version("14.0.0"):
        subprocess.run([sys.executable, "-m", "pip", "install", "rich==13.9.4"], check=True)
        st.warning("Adjusted rich to 13.9.4 for compatibility. Rerunning…")
        st.experimental_rerun()
except Exception:
    pass


# pages/03_Dashboard.py
import streamlit as st
import pandas as pd
from pymongo import MongoClient
from datetime import datetime, date, timedelta

# Try interactive charts with Plotly (fallback to Streamlit charts if not installed)
PLOTLY = True
try:
    import plotly.express as px
except Exception:
    PLOTLY = False

# ----------------------------
# MongoDB
# ----------------------------
MONGO_URI = st.secrets["mongo_uri"]
client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=8000)
db = client["TAK_DB"]
col_it = db["itineraries"]
col_up = db["package_updates"]
col_ex = db["expenses"]

# ----------------------------
# Helpers
# ----------------------------
def to_int_money(x):
    if x is None:
        return 0
    if isinstance(x, (int, float)):
        try:
            return int(round(x))
        except Exception:
            return 0
    s = str(x)
    digits = "".join(ch for ch in s if ch.isdigit())
    return int(digits) if digits else 0

def norm_date(val):
    """Return date() or None."""
    try:
        if pd.isna(val) or val is None:
            return None
        return pd.to_datetime(val).date()
    except Exception:
        return None

def load_data():
    it = list(col_it.find({}))
    up = list(col_up.find({}))
    ex = list(col_ex.find({}))

    # Itineraries
    for r in it:
        r["itinerary_id"] = str(r["_id"])
        r["upload_date"]  = norm_date(r.get("upload_date"))
        r["start_date"]   = norm_date(r.get("start_date"))
        r["end_date"]     = norm_date(r.get("end_date"))
        r["package_cost_num"] = to_int_money(r.get("package_cost"))
        r["representative"] = r.get("representative", "")  # if you saved it from app.py
        r["client_mobile"] = r.get("client_mobile", "")

    # Updates
    for r in up:
        r["booking_date"]  = norm_date(r.get("booking_date"))
        r["advance_amount"] = int(r.get("advance_amount", 0))
        r["_id"] = None  # drop mongo id

    # Expenses
    for r in ex:
        r["_id"] = None

    df_it = pd.DataFrame(it) if it else pd.DataFrame()
    df_up = pd.DataFrame(up) if up else pd.DataFrame(columns=["itinerary_id","status","booking_date","advance_amount"])
    df_ex = pd.DataFrame(ex) if ex else pd.DataFrame(columns=["itinerary_id","package_cost","total_expenses","profit"])

    # Merge
    df = df_it.merge(df_up, on="itinerary_id", how="left", suffixes=("","_up"))
    df = df.merge(df_ex, on="itinerary_id", how="left", suffixes=("","_ex"))
    # Default status
    if "status" not in df or df["status"].isna().all():
        df["status"] = "pending"
    df["status"] = df["status"].fillna("pending")

    # Latest numeric cost for analytics (prefer saved expenses package_cost)
    df["package_cost_latest"] = df["package_cost"].combine_first(df["package_cost_ex"])
    df["package_cost_latest"] = df["package_cost_latest"].apply(to_int_money)

    # Profit/expenses numeric
    df["total_expenses"] = pd.to_numeric(df["total_expenses"], errors="coerce").fillna(0).astype(int)
    df["profit"] = pd.to_numeric(df["profit"], errors="coerce").fillna(0).astype(int)

    # Represent missing cols
    for c in ["client_name","final_route","total_pax","representative"]:
        if c not in df.columns:
            df[c] = ""
    df["total_pax"] = pd.to_numeric(df["total_pax"], errors="coerce").fillna(0).astype(int)

    return df

# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="Dashboard", layout="wide")
st.title("📊 TAK – Packages Dashboard")

df_all = load_data()
if df_all.empty:
    st.info("No data yet. Upload packages in the main app.")
    st.stop()

# ----------------------------
# Filters
# ----------------------------
with st.container():
    c1, c2, c3, c4 = st.columns([1.1,1.1,1.2,2.6])
    with c1:
        basis = st.selectbox(
            "Date basis",
            ["Upload date", "Booking date", "Travel start date"],
            help="Which date to use for filtering and time-based views."
        )
    with c2:
        # Quick presets
        preset = st.selectbox("Quick range", ["This month", "Last month", "This FY", "This year", "Custom"])
    with c3:
        today = date.today()
        if preset == "This month":
            start = today.replace(day=1)
            end = (start + pd.offsets.MonthEnd(1)).date()
        elif preset == "Last month":
            first_this = today.replace(day=1)
            last_prev = (first_this - pd.offsets.Day(1)).date()
            start = last_prev.replace(day=1)
            end = last_prev
        elif preset == "This FY":
            fy_start = date(today.year if today.month>=4 else today.year-1, 4, 1)
            start, end = fy_start, today
        elif preset == "This year":
            start = date(today.year,1,1); end = today
        else:
            # default to last 90 days if custom not set yet
            start = today - timedelta(days=90); end = today
        daterange = st.date_input("Date range", (start, end))
        if isinstance(daterange, tuple) and len(daterange)==2:
            start, end = daterange
    with c4:
        reps = sorted([r for r in df_all["representative"].dropna().unique().tolist() if r])
        rep_filter = st.multiselect("Representative", options=reps, default=reps, help="Blank = unassigned will be included automatically if you uncheck all.")

# Pick the date column
date_col = {
    "Upload date": "upload_date",
    "Booking date": "booking_date",
    "Travel start date": "start_date",
}[basis]

df = df_all.copy()
# Filter by date
df[date_col] = df[date_col].apply(norm_date)
mask_date = df[date_col].between(start, end)
df = df[mask_date]

# Filter by representative
if rep_filter:
    df = df[df["representative"].isin(rep_filter)]
# else leave all (including blanks)

# Convenience flags
is_confirmed = df["status"].eq("confirmed")
is_cancelled = df["status"].eq("cancelled")
is_under_discussion = df["status"].eq("under_discussion")
is_pending = df["status"].eq("pending")
is_enquiry = is_pending | is_under_discussion  # your definition

# ----------------------------
# KPI Row
# ----------------------------
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("✅ Confirmed", int(is_confirmed.sum()))
k2.metric("🟡 Enquiries (Pending + Under discussion)", int(is_enquiry.sum()))
k3.metric("🟠 Under discussion", int(is_under_discussion.sum()))
k4.metric("🔴 Cancelled", int(is_cancelled.sum()))
# Pending cost calc = confirmed without expenses record
have_expense_ids = set(df.loc[df["total_expenses"]>0, "itinerary_id"]) | set(df.loc[df["profit"]!=0, "itinerary_id"])
pending_cost_calc = is_confirmed.sum() - len(set(df.loc[is_confirmed, "itinerary_id"]) & have_expense_ids)
k5.metric("🧾 Expense entry pending", int(max(pending_cost_calc,0)))

st.divider()

# ----------------------------
# Donut: Confirmed vs Enquiries
# ----------------------------
donut_df = pd.DataFrame({
    "Type": ["Confirmed", "Enquiries"],
    "Count": [int(is_confirmed.sum()), int(is_enquiry.sum())]
})
donut_df["Percent"] = (donut_df["Count"] / max(donut_df["Count"].sum(),1) * 100).round(1)

st.subheader("Confirmed vs Enquiries")
if PLOTLY:
    fig = px.pie(
        donut_df, values="Count", names="Type", hole=0.55,
        hover_data=["Percent"], labels={"Percent":"%"}
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.write(donut_df)
st.caption(f"Totals — Confirmed: {donut_df.loc[donut_df['Type']=='Confirmed','Count'].iat[0]} | Enquiries: {donut_df.loc[donut_df['Type']=='Enquiries','Count'].iat[0]}")

st.divider()

# ----------------------------
# Time cards: Today / Yesterday enquiries
# ----------------------------
today = date.today()
yesterday = today - timedelta(days=1)

# Enquiries based on Upload date (how many packages we created)
df_upload = df_all.copy()
df_upload["upload_date"] = df_upload["upload_date"].apply(norm_date)
enq_today = ((df_upload["upload_date"]==today) & (df_upload["status"].isin(["pending","under_discussion"]))).sum()
enq_yday  = ((df_upload["upload_date"]==yesterday) & (df_upload["status"].isin(["pending","under_discussion"]))).sum()

c1, c2 = st.columns(2)
c1.metric("📥 Enquiries created today", int(enq_today))
c2.metric("📥 Enquiries created yesterday", int(enq_yday))

st.divider()

# ----------------------------
# Totals for confirmed packages (within filter)
# ----------------------------
sum_package_cost = int(df.loc[is_confirmed, "package_cost_latest"].sum())
sum_expenses     = int(df.loc[is_confirmed, "total_expenses"].sum())
sum_profit       = int(df.loc[is_confirmed, "profit"].sum())

st.subheader("Money – Confirmed (in filter)")
m1, m2, m3 = st.columns(3)
m1.metric("💼 Total package cost (₹)", f"{sum_package_cost:,}")
m2.metric("🧾 Total expenses (₹)", f"{sum_expenses:,}")
m3.metric("💰 Total profit (₹)", f"{sum_profit:,}")

st.divider()

# ----------------------------
# Packages under discussion (visual)
# ----------------------------
st.subheader("Packages under discussion")
df_ud = df[is_under_discussion].copy()
if df_ud.empty:
    st.info("No packages under discussion in the selected range.")
else:
    # Count by representative
    rep_ct = df_ud.groupby(df_ud["representative"].replace("", "Unassigned"))["itinerary_id"].nunique().reset_index()
    rep_ct.columns = ["Representative", "Packages"]

    if PLOTLY:
        fig = px.bar(rep_ct, x="Representative", y="Packages", text="Packages")
        fig.update_traces(textposition="outside")
        fig.update_layout(yaxis_title=None, xaxis_title=None)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.bar_chart(rep_ct.set_index("Representative"))

st.divider()

# ----------------------------
# Upcoming clients table (ascending by date)
#   Basis:
#     - If basis == Booking date → show confirmed with upcoming booking_date
#     - If basis == Travel start date → show those with start_date >= today
#     - If basis == Upload date → just show newest uploads in range ascending
# ----------------------------
st.subheader("📅 Upcoming clients")
if basis == "Booking date":
    upcoming = df[is_confirmed & df["booking_date"].notna() & (df["booking_date"] >= today)].copy()
    upcoming = upcoming.sort_values("booking_date")
elif basis == "Travel start date":
    upcoming = df[df["start_date"].notna() & (df["start_date"] >= today)].copy()
    upcoming = upcoming.sort_values("start_date")
else:
    upcoming = df[df["upload_date"].notna()].copy().sort_values("upload_date")

show_cols = ["itinerary_id","client_name","representative","final_route","total_pax","package_cost_latest","status","booking_date","start_date","end_date"]
for c in show_cols:
    if c not in upcoming.columns:
        upcoming[c] = None
st.dataframe(upcoming[show_cols], use_container_width=True)

st.divider()

# ----------------------------
# Pending to confirm by representative
# ----------------------------
st.subheader("⏳ Pending to confirm (by representative)")
pending_df = df[df["status"].isin(["pending","under_discussion"])].copy()
if pending_df.empty:
    st.info("No pending items right now.")
else:
    gp = pending_df.groupby(pending_df["representative"].replace("", "Unassigned"))["itinerary_id"].nunique().reset_index()
    gp.columns = ["Representative", "Pending"]
    if PLOTLY:
        fig = px.bar(gp, x="Representative", y="Pending", text="Pending")
        fig.update_traces(textposition="outside")
        fig.update_layout(yaxis_title=None, xaxis_title=None)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.bar_chart(gp.set_index("Representative"))

# ----------------------------
# (Optional) Extras you can add later:
#  - Trend over time (confirmed/enquiries by week)
#  - Top profit packages
#  - Representative conversion rate (confirmed / enquiries)
# ----------------------------

