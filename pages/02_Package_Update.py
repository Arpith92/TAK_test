# pages/02_Package_Update.py
import os
import streamlit as st
import pandas as pd
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError
from datetime import datetime, date
from bson import ObjectId

# Optional: pretty calendar
CALENDAR_AVAILABLE = True
try:
    from streamlit_calendar import calendar
except Exception:
    CALENDAR_AVAILABLE = False

# ----------------------------
# Page config (early, so UI renders even if DB fails)
# ----------------------------
st.set_page_config(page_title="Package Update", layout="wide")
st.title("📦 Package Update")

# ----------------------------
# Safe secrets + Mongo helper
# ----------------------------
@st.cache_resource
def get_db():
    """
    Returns a connected DB handle or shows a clear UI error and stops.
    Looks for Streamlit secret 'mongo_uri' first, then env var MONGO_URI.
    """
    uri = (st.secrets.get("mongo_uri") if hasattr(st, "secrets") else None) or os.getenv("MONGO_URI")
    if not uri:
        st.error(
            "Mongo connection is not configured.\n\n"
            "Add `mongo_uri` in **Manage app → Settings → Secrets** for this app.\n"
            "Example:\n"
            'mongo_uri = "mongodb+srv://USER:PASS@host/?retryWrites=true&w=majority"\n\n'
            "Tip: Secrets are per-app. Ensure you added them to *this* app."
        )
        try:
            # Non-sensitive heads-up for debugging: which secret keys exist
            st.caption(f"Current secret keys: {list(st.secrets.keys())}")
        except Exception:
            pass
        st.stop()

    client = MongoClient(uri, serverSelectionTimeoutMS=8000)
    try:
        client.admin.command("ping")
    except ServerSelectionTimeoutError as e:
        st.error(f"Could not connect to MongoDB. Please verify the URI and IP allowlist.\n\nDetails: {e}")
        st.stop()
    return client["TAK_DB"]

# Create DB + collections
db = get_db()
col_itineraries = db["itineraries"]          # created by app.py
col_updates     = db["package_updates"]      # status + booking_date + advance_amount
col_expenses    = db["expenses"]             # vendor costs + totals

# (Optional) read login users (won't crash if missing)
USERS = dict(st.secrets.get("users", {})) if hasattr(st, "secrets") else {}
if not USERS:
    st.warning("Login is not configured yet. Add a [users] block in Secrets if your app uses it.")

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

def current_fy_two_digits(today: date | None = None) -> int:
    """Financial year (India style): Apr–Mar. Return last two digits (e.g., 25)."""
    d = today or date.today()
    year = d.year if d.month >= 4 else d.year - 1
    return year % 100

def next_ach_id(fy_2d: int) -> str:
    """Find next sequence for ACH-<fy>-NNN based on existing docs with ach_id."""
    prefix = f"ACH-{fy_2d:02d}-"
    docs = col_itineraries.find({"ach_id": {"$regex": f"^{prefix}\\d{{3}}$"}}, {"ach_id": 1})
    max_no = 0
    for d in docs:
        try:
            n = int(d["ach_id"].split("-")[-1])
            if n > max_no:
                max_no = n
        except Exception:
            pass
    return f"{prefix}{max_no+1:03d}"

def backfill_ach_ids():
    """Give ACH IDs to itineraries that don't have one yet (one-time per doc)."""
    fy = current_fy_two_digits()
    cursor = col_itineraries.find({"$or": [{"ach_id": {"$exists": False}}, {"ach_id": ""}]})
    for doc in cursor:
        new_id = next_ach_id(fy)
        col_itineraries.update_one({"_id": doc["_id"]}, {"$set": {"ach_id": new_id}})

def fetch_itineraries_df():
    backfill_ach_ids()
    rows = list(col_itineraries.find({}))
    if not rows:
        return pd.DataFrame()
    for r in rows:
        r["itinerary_id"] = str(r.get("_id"))
        r["ach_id"] = r.get("ach_id", "")
        # normalize dates
        for k in ("start_date", "end_date"):
            try:
                v = r.get(k)
                r[k] = pd.to_datetime(v).date() if pd.notna(v) else None
            except Exception:
                r[k] = None
        r["package_cost_num"] = to_int_money(r.get("package_cost"))
    return pd.DataFrame(rows)

def fetch_updates_df():
    rows = list(col_updates.find({}, {"_id": 0}))
    if not rows:
        return pd.DataFrame(columns=["itinerary_id","status","booking_date","advance_amount"])
    for r in rows:
        if r.get("booking_date"):
            try:
                r["booking_date"] = pd.to_datetime(r["booking_date"]).date()
            except Exception:
                r["booking_date"] = None
        r["advance_amount"] = to_int_money(r.get("advance_amount", 0))
    return pd.DataFrame(rows)

def fetch_expenses_df():
    rows = list(col_expenses.find({}, {"_id":0}))
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["itinerary_id","total_expenses","profit","package_cost"])

def upsert_status(itinerary_id, status, booking_date, advance_amount):
    doc = {
        "itinerary_id": itinerary_id,
        "status": status,
        "updated_at": datetime.utcnow(),
        "advance_amount": int(advance_amount or 0)
    }
    if status == "confirmed" and booking_date:
        doc["booking_date"] = booking_date
    else:
        doc["booking_date"] = None
    col_updates.update_one({"itinerary_id": itinerary_id}, {"$set": doc}, upsert=True)

def save_expenses(itinerary_id, client_name, booking_date, package_cost, vendors, notes=""):
    total_expenses = sum(int(v or 0) for _, v in vendors)
    profit = int(package_cost) - int(total_expenses)
    doc = {
        "itinerary_id": itinerary_id,
        "client_name": client_name,
        "booking_date": booking_date,
        "package_cost": int(package_cost),   # overwrite with latest
        "total_expenses": int(total_expenses),
        "profit": int(profit),
        "vendors": [{"name": n, "cost": int(v or 0)} for n, v in vendors],
        "notes": notes,
        "saved_at": datetime.utcnow(),
    }
    col_expenses.update_one({"itinerary_id": itinerary_id}, {"$set": doc}, upsert=True)
    return profit

def find_itinerary_doc(selected_id: str):
    """Fetch itinerary doc by _id (ObjectId), or fallback to string fields."""
    it = None
    try:
        it = col_itineraries.find_one({"_id": ObjectId(selected_id)})
    except Exception:
        it = None
    if it is None:
        it = (col_itineraries.find_one({"itinerary_id": selected_id}) or
              col_itineraries.find_one({"ach_id": selected_id}))
    return it

def to_date_or_none(x):
    try:
        if pd.isna(x):
            return None
        return pd.to_datetime(x).date()
    except Exception:
        return None

# ----------------------------
# Load data
# ----------------------------
df_it = fetch_itineraries_df()
if df_it.empty:
    st.info("No packages found yet. Upload a file in the main app first.")
    st.stop()

df_up  = fetch_updates_df()
df_exp = fetch_expenses_df()

# (fixed minor typo: use on="itinerary_id")
df = df_it.merge(df_up, on="itinerary_id", how="left")
df["status"] = df["status"].fillna("pending")

# Ensure advance_amount exists & is numeric
if "advance_amount" not in df.columns:
    df["advance_amount"] = 0
df["advance_amount"] = pd.to_numeric(df["advance_amount"], errors="coerce").fillna(0).astype(int)

# Normalize date columns to avoid NaT in editor
for col in ["start_date", "end_date", "booking_date"]:
    if col in df.columns:
        df[col] = df[col].apply(to_date_or_none)

# ----------------------------
# Summary KPIs
# ----------------------------
pending_count           = (df["status"] == "pending").sum()
under_discussion_count  = (df["status"] == "under_discussion").sum()
cancelled_count         = (df["status"] == "cancelled").sum()

confirmed = df[df["status"] == "confirmed"].copy()
have_expense_ids = set(df_exp["itinerary_id"]) if not df_exp.empty else set()
confirmed_expense_pending = confirmed[~confirmed["itinerary_id"].isin(have_expense_ids)].shape[0]

k1, k2, k3, k4 = st.columns(4)
k1.metric("🟡 Pending", int(pending_count))
k2.metric("🟠 Under discussion", int(under_discussion_count))
k3.metric("🟧 Confirmed – expense pending", int(confirmed_expense_pending))
k4.metric("🔴 Cancelled", int(cancelled_count))

st.divider()

# ----------------------------
# 1) Status Update (with bulk action)
# ----------------------------
st.subheader("1) Update Status for Pending / Under Discussion")

editable = df[df["status"].isin(["pending", "under_discussion"])].copy()

if editable.empty:
    st.success("No pending or under-discussion packages right now. 🎉")
else:
    # guarantee required columns exist
    must_cols = [
        "ach_id","itinerary_id","client_name","final_route","total_pax",
        "start_date","end_date","package_cost","status","booking_date","advance_amount"
    ]
    for c in must_cols:
        if c not in editable.columns:
            editable[c] = None

    # normalize dtypes (critical for data_editor)
    for c in ["ach_id","itinerary_id","client_name","final_route","package_cost"]:
        editable[c] = editable[c].fillna("").astype(str)
    editable["total_pax"] = pd.to_numeric(editable["total_pax"], errors="coerce").fillna(0).astype(int)
    editable["advance_amount"] = pd.to_numeric(editable["advance_amount"], errors="coerce").fillna(0).astype(int)
    for c in ["start_date","end_date","booking_date"]:
        editable[c] = editable[c].apply(to_date_or_none)

    show_cols = [
        "ach_id","itinerary_id","client_name","final_route","total_pax",
        "start_date","end_date","package_cost","status","booking_date","advance_amount"
    ]
    editable = editable[show_cols].sort_values(["start_date","client_name"], na_position="last")

    st.caption("Tip: Select rows below and use the **Bulk update** box to update all selected at once.")

    # Some Streamlit versions don't support selection_mode; fall back if needed
    try:
        edited = st.data_editor(
            editable,
            use_container_width=True,
            hide_index=True,
            selection_mode="multi-row",
            column_config={
                "status": st.column_config.SelectboxColumn(
                    "Status", options=["pending","under_discussion","confirmed","cancelled"]
                ),
                "booking_date": st.column_config.DateColumn("Booking date", format="YYYY-MM-DD"),
                "advance_amount": st.column_config.NumberColumn("Advance (₹)", min_value=0, step=500),
            },
            key="status_editor"
        )
        selection_supported = True
    except TypeError:
        # fallback without selection_mode
        edited = st.data_editor(
            editable,
            use_container_width=True,
            hide_index=True,
            column_config={
                "status": st.column_config.SelectboxColumn(
                    "Status", options=["pending","under_discussion","confirmed","cancelled"]
                ),
                "booking_date": st.column_config.DateColumn("Booking date", format="YYYY-MM-DD"),
                "advance_amount": st.column_config.NumberColumn("Advance (₹)", min_value=0, step=500),
            },
            key="status_editor"
        )
        selection_supported = False

    sel = []
    if selection_supported:
        sel = st.session_state.get("status_editor", {}).get("selection", {}).get("rows", []) or []

    with st.expander("🔁 Bulk update selected rows"):
        bcol1, bcol2, bcol3, bcol4 = st.columns([1,1,1,1])
        with bcol1:
            bulk_status = st.selectbox("Status", ["pending","under_discussion","confirmed","cancelled"])
        with bcol2:
            bulk_date = st.date_input("Booking date (for confirmed)", value=None)
        with bcol3:
            bulk_adv = st.number_input("Advance (₹)", min_value=0, step=500, value=0)
        with bcol4:
            apply_bulk = st.button("Apply to selected", disabled=not selection_supported)

        if not selection_supported:
            st.caption("Multi-row selection not supported in this Streamlit build; edit rows below, then click **Save row-by-row edits**.")
        elif apply_bulk:
            if not sel:
                st.warning("No rows selected.")
            else:
                for r_idx in sel:
                    r = edited.iloc[r_idx]
                    bdate = None
                    if bulk_status == "confirmed":
                        if not bulk_date:
                            continue
                        bdate = pd.to_datetime(bulk_date).date().isoformat()
                    upsert_status(r["itinerary_id"], bulk_status, bdate, bulk_adv)
                st.success(f"Applied to {len(sel)} row(s).")
                st.rerun()

    if st.button("💾 Save row-by-row edits"):
        saved, errors = 0, 0
        for _, r in edited.iterrows():
            itinerary_id = r["itinerary_id"]
            status = r["status"]
            bdate = r.get("booking_date")
            adv   = r.get("advance_amount", 0)
            if status == "confirmed":
                if bdate is None or (isinstance(bdate, str) and not bdate):
                    errors += 1
                    continue
                bdate = pd.to_datetime(bdate).date().isoformat()
            else:
                bdate = None
            try:
                upsert_status(itinerary_id, status, bdate, adv)
                saved += 1
            except Exception:
                errors += 1
        if saved:
            st.success(f"Saved {saved} update(s).")
        if errors:
            st.warning(f"{errors} row(s) skipped (missing/invalid booking date for confirmed).")
        st.rerun()

st.divider()

# ----------------------------
# 2) Expense Entry for Confirmed Packages
# ----------------------------
st.subheader("2) Enter Expenses for Confirmed Packages")

df_up = fetch_updates_df()
df = df_it.merge(df_up, on="itinerary_id", how="left")
df["status"] = df["status"].fillna("pending")
if "advance_amount" not in df.columns:
    df["advance_amount"] = 0
df["advance_amount"] = pd.to_numeric(df["advance_amount"], errors="coerce").fillna(0).astype(int)
for c in ["booking_date", "start_date", "end_date"]:
    if c in df.columns:
        df[c] = df[c].apply(to_date_or_none)

confirmed = df[df["status"] == "confirmed"].copy()

if confirmed.empty:
    st.info("No confirmed packages yet.")
else:
    have_expense = set(df_exp["itinerary_id"]) if not df_exp.empty else set()
    confirmed["expense_entered"] = confirmed["itinerary_id"].isin(have_expense)

    left, right = st.columns([2,1])
    with left:
        show_cols = ["ach_id","itinerary_id","client_name","final_route","total_pax",
                     "package_cost","advance_amount","booking_date","expense_entered"]
        st.dataframe(
            confirmed[show_cols].sort_values("booking_date"),
            use_container_width=True
        )
    with right:
        st.markdown("**Select a confirmed package to add/edit expenses:**")
        options = (confirmed["ach_id"].fillna("") + " | " +
                   confirmed["client_name"].fillna("") + " | " +
                   confirmed["booking_date"].fillna("").astype(str) + " | " +
                   confirmed["itinerary_id"])
        sel = st.selectbox("Choose package", options.tolist() if not options.empty else [])
        chosen_id = sel.split(" | ")[-1] if sel else None

    if chosen_id:
        row = confirmed[confirmed["itinerary_id"] == chosen_id].iloc[0]
        client_name  = row.get("client_name","")
        booking_date = row.get("booking_date","")
        base_cost = st.number_input(
            "Package cost (₹)",
            min_value=0,
            value=to_int_money(row.get("package_cost") or row.get("package_cost_num")),
            step=500
        )

        st.markdown(f"**Client:** {client_name}  \n**Booking date:** {booking_date}")
        st.markdown("#### Expense Inputs")

        with st.form("expense_form"):
            auto_vendor = st.text_input("Auto Vendor Name")
            auto_cost   = st.number_input("Auto Cost (₹)", min_value=0, step=100)

            c1, c2 = st.columns(2)
            with c1:
                car_vendor_1 = st.text_input("Car vendor Name-1")
                car_cost_1   = st.number_input("Car Cost-1 (₹)", min_value=0, step=100)
                hotel_vendor_1 = st.text_input("Hotel vendor-1")
                hotel_cost_1   = st.number_input("Hotel vendor-1 cost (₹)", min_value=0, step=100)
                hotel_vendor_3 = st.text_input("Hotel vendor-3")
                hotel_cost_3   = st.number_input("Hotel vendor-3 cost (₹)", min_value=0, step=100)
                hotel_vendor_5 = st.text_input("Hotel vendor-5")
                hotel_cost_5   = st.number_input("Hotel vendor-5 cost (₹)", min_value=0, step=100)
            with c2:
                car_vendor_2 = st.text_input("Car vendor Name-2")
                car_cost_2   = st.number_input("Car Cost-2 (₹)", min_value=0, step=100)
                hotel_vendor_2 = st.text_input("Hotel vendor-2")
                hotel_cost_2   = st.number_input("Hotel vendor-2 cost (₹)", min_value=0, step=100)
                hotel_vendor_4 = st.text_input("Hotel vendor-4")
                hotel_cost_4   = st.number_input("Hotel vendor-4 cost (₹)", min_value=0, step=100)
                other_pooja_vendor = st.text_input("Other Pooja vendor")
                other_pooja_cost   = st.number_input("Other Pooja vendor cost (₹)", min_value=0, step=100)

            bhas_vendor = st.text_input("Bhasmarathi vendor")
            bhas_cost   = st.number_input("Bhasmarathi Cost (₹)", min_value=0, step=100)

            photo_cost  = st.number_input("Photo frame cost (₹)", min_value=0, step=100)
            other_exp   = st.number_input("Any other expense (₹)", min_value=0, step=100)
            notes       = st.text_area("Notes (optional)")

            submit = st.form_submit_button("💾 Save Expenses")

        if submit:
            vendors = [
                ("Auto", auto_cost),
                (f"Car-1 | {car_vendor_1}", car_cost_1),
                (f"Car-2 | {car_vendor_2}", car_cost_2),
                (f"Hotel-1 | {hotel_vendor_1}", hotel_cost_1),
                (f"Hotel-2 | {hotel_vendor_2}", hotel_cost_2),
                (f"Hotel-3 | {hotel_vendor_3}", hotel_cost_3),
                (f"Hotel-4 | {hotel_vendor_4}", hotel_cost_4),
                (f"Hotel-5 | {hotel_vendor_5}", hotel_cost_5),
                (f"Bhasmarathi | {bhas_vendor}", bhas_cost),
                (f"Other Pooja | {other_pooja_vendor}", other_pooja_cost),
                ("Photo frame", photo_cost),
                ("Other", other_exp),
            ]
            profit = save_expenses(chosen_id, client_name, booking_date, base_cost, vendors, notes)
            st.success(f"Expenses saved. 💰 Profit: ₹ {profit:,}")
            st.rerun()

st.divider()

# ----------------------------
# 3) Calendar – Toggle views & click-to-open details
# ----------------------------
st.subheader("3) Calendar – Confirmed Packages")
view = st.radio("View", ["By Booking Date", "By Travel Dates"], horizontal=True)

confirmed = df[df["status"] == "confirmed"].copy()
if confirmed.empty:
    st.info("No confirmed packages to show on calendar.")
else:
    events = []
    for _, r in confirmed.iterrows():
        title = f"{r.get('client_name','')}_{r.get('total_pax','')}pax"
        ev = {"title": title, "id": r["itinerary_id"]}

        if view == "By Booking Date":
            if pd.isna(r.get("booking_date")):
                continue
            ev["start"] = pd.to_datetime(r["booking_date"]).strftime("%Y-%m-%d")
        else:
            if pd.isna(r.get("start_date")) or pd.isna(r.get("end_date")):
                continue
            ev["start"] = pd.to_datetime(r["start_date"]).strftime("%Y-%m-%d")
            # FullCalendar uses exclusive end -> add one day to show a bar across dates
            end_ = pd.to_datetime(r["end_date"]) + pd.Timedelta(days=1)
            ev["end"] = end_.strftime("%Y-%m-%d")

        events.append(ev)

    selected_id = None
    if CALENDAR_AVAILABLE:
        opts = {"initialView": "dayGridMonth", "height": 620, "eventDisplay": "block"}
        result = calendar(options=opts, events=events, key=f"pkg_cal_{'booking' if view=='By Booking Date' else 'travel'}")
        if result and isinstance(result, dict) and result.get("eventClick"):
            try:
                selected_id = result["eventClick"]["event"]["id"]
            except Exception:
                selected_id = None
    else:
        st.caption("Calendar component not installed. Showing a simple list instead.")
        display = pd.DataFrame(events).rename(columns={"title": "Package", "start": "Start", "end": "End"})
        st.dataframe(display.sort_values(["Start","End"]), use_container_width=True)
        if not confirmed.empty:
            selected_id = st.selectbox(
                "Open package details",
                (confirmed["itinerary_id"] + " | " + confirmed["client_name"]).tolist()
            )
            if selected_id:
                selected_id = selected_id.split(" | ")[0]

    if selected_id:
        st.divider()
        st.subheader("📦 Package Details")

        it = find_itinerary_doc(selected_id)
        upd = col_updates.find_one({"itinerary_id": selected_id}, {"_id":0})
        exp = col_expenses.find_one({"itinerary_id": selected_id}, {"_id":0})

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Basic**")
            st.write({
                "ACH ID": it.get("ach_id", "") if it else "",
                "Client": it.get("client_name","") if it else "",
                "Route": it.get("final_route","") if it else "",
                "Pax": it.get("total_pax","") if it else "",
                "Travel": f"{it.get('start_date','')} → {it.get('end_date','')}" if it else "",
            })
        with c2:
            st.markdown("**Status & Money**")
            st.write({
                "Status": upd.get("status","") if upd else "",
                "Booking date": upd.get("booking_date","") if upd else "",
                "Advance (₹)": upd.get("advance_amount",0) if upd else 0,
                "Package cost (₹)": (exp.get("package_cost")
                                     if exp and "package_cost" in exp
                                     else to_int_money(it.get("package_cost")) if it else 0),
                "Total expenses (₹)": exp.get("total_expenses", 0) if exp else 0,
                "Profit (₹)": exp.get("profit", 0) if exp else 0,
            })

        st.markdown("**Itinerary text**")
        st.text_area("Shared with client", value=(it.get("itinerary_text","") if it else ""), height=260, disabled=True)
