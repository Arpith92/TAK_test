# --- Optional safety: adjust rich if Cloud ever downgrades Streamlit later ---
try:
    import streamlit as st, rich
    from packaging.version import Version
    import sys, subprocess
    if Version(st.__version__) < Version("1.42.0") and Version(rich.__version__) >= Version("14.0.0"):
        subprocess.run([sys.executable, "-m", "pip", "install", "rich==13.9.4"], check=True)
        st.warning("Adjusted rich to 13.9.4 for compatibility. Rerunning…")
        st.experimental_rerun()
except Exception:
    pass

# pages/03_Dashboard.py
from datetime import datetime, date, timedelta
import pandas as pd
import streamlit as st
from pymongo import MongoClient

# Try interactive charts with Plotly (fallback to Streamlit charts if not installed)
PLOTLY = True
try:
    import plotly.express as px
except Exception:
    PLOTLY = False

# ----------------------------
# Admin gate (same password style as other admin pages)
# ----------------------------
def require_admin():
    ADMIN_PASS_DEFAULT = "Arpith&92--"             # same default
    ADMIN_PASS = str(st.secrets.get("admin_pass", ADMIN_PASS_DEFAULT))

    with st.sidebar:
        st.markdown("### Admin access")
        p = st.text_input("Enter admin password", type="password", placeholder="Arpith&92--")
    if (p or "").strip() != ADMIN_PASS.strip():
        st.stop()
    st.session_state["user"] = "Admin"
    st.session_state["is_admin"] = True

# ----------------------------
# MongoDB
# ----------------------------
MONGO_URI = st.secrets["mongo_uri"]
client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=8000)
db = client["TAK_DB"]
col_it = db["itineraries"]
col_up = db["package_updates"]
col_ex = db["expenses"]
col_fu = db["followups"]   # NEW: follow-up logs

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

def month_bounds(d: date):
    first = d.replace(day=1)
    # last day of month trick
    last = (first + pd.offsets.MonthEnd(1)).date()
    return first, last

def fy_bounds(d: date):
    start = date(d.year if d.month >= 4 else d.year - 1, 4, 1)
    return start, d

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
        r["_id"] = None

    # Updates
    for r in up:
        r["booking_date"]   = norm_date(r.get("booking_date"))
        r["advance_amount"] = int(r.get("advance_amount", 0))
        r["_id"] = None

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
    for c in ["client_name","final_route","total_pax","representative","client_mobile"]:
        if c not in df.columns:
            df[c] = ""
    df["total_pax"] = pd.to_numeric(df["total_pax"], errors="coerce").fillna(0).astype(int)

    return df

# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="Dashboard", layout="wide")
st.title("📊 TAK – Packages Dashboard")
require_admin()

# Top bar: search (top-right) + quick open Follow-up view
tb1, tb2, tb3 = st.columns([4,2,2])
with tb2:
    followup_view = st.button("📞 followup_update", help="View activities logged in Follow-up page")
with tb3:
    search_txt = st.text_input("Search", placeholder="Client name / Mobile", label_visibility="collapsed")

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
            start, end = month_bounds(today)
        elif preset == "Last month":
            first_this, _ = month_bounds(today)
            last_prev = first_this - timedelta(days=1)
            start, end = month_bounds(last_prev)
        elif preset == "This FY":
            start, end = fy_bounds(today)
        elif preset == "This year":
            start, end = date(today.year,1,1), today
        else:
            start, end = today - timedelta(days=90), today
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

# Apply search (client name / mobile)
if (search_txt or "").strip():
    s = search_txt.strip().lower()
    df = df[
        df["client_name"].astype(str).str.lower().str.contains(s) |
        df["client_mobile"].astype(str).str.lower().str.contains(s)
    ]

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
# Time cards: Today / Yesterday enquiries (based on original uploads)
# ----------------------------
today = date.today()
yesterday = today - timedelta(days=1)
df_upload = df_all.copy()
df_upload["upload_date"] = df_upload["upload_date"].apply(norm_date)
enq_today = ((df_upload["upload_date"]==today) & (df_upload["status"].isin(["pending","under_discussion"]))).sum()
enq_yday  = ((df_upload["upload_date"]==yesterday) & (df_upload["status"].isin(["pending","under_discussion"]))).sum()

c1, c2 = st.columns(2)
c1.metric("📥 Enquiries created today", int(enq_today))
c2.metric("📥 Enquiries created yesterday", int(enq_yday))

st.divider()

# ----------------------------
# Money totals for confirmed (within filter)
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
# 📞 Follow-up updates (admin-only detail)
# ----------------------------
if followup_view:
    st.subheader("📞 Follow-up updates (activity log)")

    # Pull follow-up logs in the chosen date range (by created_at)
    start_dt = datetime.combine(start, datetime.min.time())
    end_dt   = datetime.combine(end,   datetime.max.time())
    cur = col_fu.find({"created_at": {"$gte": start_dt, "$lte": end_dt}})
    logs = list(cur)

    if not logs:
        st.info("No follow-up activity in the selected range.")
    else:
        # Normalize + search filter
        rows = []
        for d in logs:
            rows.append({
                "itinerary_id": str(d.get("itinerary_id","")),
                "ach_id": d.get("ach_id",""),
                "client_name": d.get("client_name",""),
                "client_mobile": d.get("client_mobile",""),
                "created_at": d.get("created_at"),
                "created_by": d.get("created_by",""),
                "status": d.get("status",""),
                "next_followup_on": d.get("next_followup_on"),
                "comment": d.get("comment",""),
            })
        df_fu = pd.DataFrame(rows)
        if not df_fu.empty:
            df_fu["created_at"] = pd.to_datetime(df_fu["created_at"])
            df_fu["next_followup_on"] = pd.to_datetime(df_fu["next_followup_on"], errors="coerce")

            # Apply search
            if (search_txt or "").strip():
                s = search_txt.strip().lower()
                df_fu = df_fu[
                    df_fu["client_name"].astype(str).str.lower().str.contains(s) |
                    df_fu["client_mobile"].astype(str).str.lower().str.contains(s)
                ]

            # Client-wise attempt counts and confirmation flag
            attempts = (df_fu["status"]=="followup").groupby(df_fu["itinerary_id"]).sum().rename("followup_attempts")
            confirmed_flag = (df_fu["status"]=="confirmed").groupby(df_fu["itinerary_id"]).any().rename("confirmed_from_followup")
            head = df_fu.groupby("itinerary_id").first()[["ach_id","client_name","client_mobile"]]
            fu_summary = head.join([attempts, confirmed_flag]).fillna({"followup_attempts":0, "confirmed_from_followup":False}).reset_index()

            # Show KPIs
            total_attempts = int(fu_summary["followup_attempts"].sum())
            total_confirmed_from_fu = int(fu_summary["confirmed_from_followup"].sum())
            kf1, kf2 = st.columns(2)
            kf1.metric("📞 Total follow-up attempts", total_attempts)
            kf2.metric("✅ Confirmed from follow-up", total_confirmed_from_fu)

            # Table
            st.markdown("**Client-wise summary**")
            show_cols = ["ach_id","client_name","client_mobile","followup_attempts","confirmed_from_followup"]
            st.dataframe(fu_summary[show_cols].sort_values(["confirmed_from_followup","followup_attempts"], ascending=[False, False]),
                         use_container_width=True, hide_index=True)

            # Full trails (expanders)
            st.markdown("**Remark trails**")
            for iid, grp in df_fu.sort_values("created_at", ascending=False).groupby("itinerary_id"):
                meta = fu_summary.loc[fu_summary["itinerary_id"]==iid].iloc[0]
                label = f"{meta['ach_id']} — {meta['client_name']} ({meta['client_mobile']})"
                with st.expander(label, expanded=False):
                    show = grp[["created_at","created_by","status","next_followup_on","comment"]].copy()
                    show.rename(columns={
                        "created_at":"When",
                        "created_by":"By",
                        "status":"Status",
                        "next_followup_on":"Next follow-up",
                        "comment":"Comment"
                    }, inplace=True)
                    st.dataframe(show, use_container_width=True, hide_index=True)

    st.divider()

# ----------------------------
# 📅 Upcoming clients table (ascending by date)
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
# ⏳ Pending to confirm by representative
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

st.divider()

# ----------------------------
# 🎯 Incentives (all representatives)
# ----------------------------
st.subheader("🎯 Incentives overview (all reps)")

# Period selector
p1, p2, p3 = st.columns([1.2, 1.2, 2.6])
with p1:
    inc_mode = st.selectbox("Period", ["This month", "Last month", "This FY", "This year", "Custom month", "Custom range"])
with p2:
    today = date.today()
    if inc_mode == "This month":
        inc_start, inc_end = month_bounds(today)
    elif inc_mode == "Last month":
        first_this, _ = month_bounds(today)
        last_prev = first_this - timedelta(days=1)
        inc_start, inc_end = month_bounds(last_prev)
    elif inc_mode == "This FY":
        inc_start, inc_end = fy_bounds(today)
    elif inc_mode == "This year":
        inc_start, inc_end = date(today.year,1,1), today
    elif inc_mode == "Custom month":
        y = st.number_input("Year", min_value=2023, max_value=today.year, value=today.year, step=1)
        m = st.selectbox("Month", list(range(1,13)), index=today.month-1)
        inc_start = date(int(y), int(m), 1)
        inc_end = (inc_start + pd.offsets.MonthEnd(1)).date()
    else:
        inc_start, inc_end = today.replace(day=1), today
with p3:
    if inc_mode == "Custom range":
        dr = st.date_input("Choose range", (inc_start, inc_end))
        if isinstance(dr, tuple) and len(dr)==2:
            inc_start, inc_end = dr

# Pull confirmed updates in range (by booking_date) and aggregate incentives by rep
q = {
    "status": "confirmed",
    "booking_date": {
        "$gte": datetime.combine(inc_start, datetime.min.time()),
        "$lte": datetime.combine(inc_end, datetime.max.time())
    }
}
conf_upd = list(col_up.find(q, {"_id":0, "rep_name":1, "incentive":1}))
df_inc = pd.DataFrame(conf_upd)
if df_inc.empty:
    st.info("No confirmed packages in the selected period.")
else:
    df_inc["rep_name"] = df_inc["rep_name"].fillna("").replace("", "Unassigned")
    df_inc["incentive"] = pd.to_numeric(df_inc["incentive"], errors="coerce").fillna(0).astype(int)
    # Count confirmed and sum incentives
    rep_summary = df_inc.groupby("rep_name").agg(
        confirmed=("incentive","count"),
        incentive=("incentive","sum")
    ).reset_index().sort_values("incentive", ascending=False)

    i1, i2 = st.columns(2)
    i1.metric("Total confirmed (period)", int(rep_summary["confirmed"].sum()))
    i2.metric("Total incentives (₹)", f"{int(rep_summary['incentive'].sum()):,}")

    st.dataframe(rep_summary.rename(columns={"rep_name":"Representative","confirmed":"Confirmed","incentive":"Incentive (₹)"}),
                 use_container_width=True, hide_index=True)

    if PLOTLY:
        fig = px.bar(rep_summary, x="rep_name", y="incentive", text="incentive", labels={"rep_name":"Representative","incentive":"Incentive (₹)"})
        fig.update_traces(textposition="outside")
        fig.update_layout(yaxis_title=None, xaxis_title=None)
        st.plotly_chart(fig, use_container_width=True)
