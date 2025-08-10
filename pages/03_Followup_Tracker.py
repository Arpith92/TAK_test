# pages/03_Followup_Tracker.py
from __future__ import annotations

from datetime import datetime, date, timedelta
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st
from bson import ObjectId
from pymongo import MongoClient

# ----------------------------
# Fallback loader for users (if Cloud Secrets miss [users])
# ----------------------------
def load_users() -> dict:
    users = st.secrets.get("users", None)
    if isinstance(users, dict) and users:
        return users
    try:
        try:
            import tomllib  # Python 3.11+
        except Exception:  # pragma: no cover
            import tomli as tomllib  # older pythons
        with open(".streamlit/secrets.toml", "rb") as f:
            data = tomllib.load(f)
        u = data.get("users", {})
        if isinstance(u, dict) and u:
            with st.sidebar:
                st.warning(
                    "Using users from repo .streamlit/secrets.toml. "
                    "For production, set them in Manage app → Secrets."
                )
            return u
    except Exception:
        pass
    return {}

# ----------------------------
# Enforced PIN login (+ Logout)
# ----------------------------
def _login() -> Optional[str]:
    with st.sidebar:
        if st.session_state.get("user"):
            st.markdown(f"**Signed in as:** {st.session_state['user']}")
            if st.button("Log out"):
                st.session_state.pop("user", None)
                st.rerun()

    if st.session_state.get("user"):
        return st.session_state["user"]

    users_map = load_users()
    if not isinstance(users_map, dict) or not users_map:
        with st.sidebar:
            st.caption("Secrets debug")
            try:
                st.write("keys:", list(st.secrets.keys()))
            except Exception:
                st.write("keys: unavailable")
            st.write("users type:", type(st.secrets.get("users", None)).__name__)
        st.error(
            "Login is not configured yet.\n\n"
            "Add to **Manage app → Secrets**:\n"
            'mongo_uri = "mongodb+srv://…"\n\n'
            "[users]\nArpith = \"1234\"\nReena = \"5678\"\nTeena = \"7777\"\nKuldeep = \"8888\"\n"
        )
        st.stop()

    st.markdown("### 🔐 Login")
    c1, c2 = st.columns([1, 1])
    with c1:
        name = st.selectbox("User", list(users_map.keys()), key="login_user")
    with c2:
        pin = st.text_input("PIN", type="password", key="login_pin")

    if st.button("Sign in"):
        if str(users_map.get(name, "")).strip() == str(pin).strip():
            st.session_state["user"] = name
            st.success(f"Welcome, {name}!")
            st.rerun()
        else:
            st.error("Invalid PIN"); st.stop()
    return None

# ----------------------------
# Mongo setup
# ----------------------------
MONGO_URI = st.secrets["mongo_uri"]
client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=8000)
db = client["TAK_DB"]

col_itineraries = db["itineraries"]
col_updates     = db["package_updates"]
col_followups   = db["followups"]
col_expenses    = db["expenses"]  # stores final package_cost etc.

# ----------------------------
# Helpers
# ----------------------------
def _to_int(x, default=0):
    try:
        if x is None:
            return default
        return int(float(str(x).replace(",", "")))
    except Exception:
        return default

def _clean_dt(x):
    if x is None:
        return None
    try:
        ts = pd.to_datetime(x)
        if isinstance(ts, pd.Timestamp):
            ts = ts.to_pydatetime()
        return ts
    except Exception:
        return None

def _today():
    return datetime.utcnow().date()

def month_bounds(d: date):
    first = d.replace(day=1)
    next_month = (first.replace(day=28) + timedelta(days=4)).replace(day=1)
    last = next_month - timedelta(days=1)
    return first, last

def _package_amount_for(iid: str) -> int:
    """Prefer saved final package_cost from expenses, else itinerary.package_cost."""
    exp = col_expenses.find_one({"itinerary_id": str(iid)}, {"package_cost": 1}) or {}
    if "package_cost" in exp:
        return _to_int(exp.get("package_cost", 0))
    it = col_itineraries.find_one({"_id": ObjectId(iid)}, {"package_cost": 1}) or {}
    return _to_int(it.get("package_cost", 0))

def _compute_incentive(pkg_amt: int) -> int:
    if 5000 < pkg_amt < 20000:
        return 250
    if pkg_amt >= 20000:
        return 500
    return 0

# ----------------------------
# Data fetchers
# ----------------------------
def fetch_assigned_followups(user: str) -> pd.DataFrame:
    rows = list(col_updates.find(
        {"status": "followup", "assigned_to": user},
        {"_id": 0}
    ))
    if not rows:
        return pd.DataFrame(columns=["itinerary_id", "assigned_to", "status"])
    df_u = pd.DataFrame(rows)
    df_u["itinerary_id"] = df_u["itinerary_id"].astype(str)

    its = list(col_itineraries.find({}, {
        "_id": 1, "ach_id": 1, "client_name": 1, "client_mobile": 1,
        "start_date": 1, "end_date": 1, "final_route": 1, "total_pax": 1,
        "representative": 1, "itinerary_text": 1
    }))
    for r in its:
        r["itinerary_id"] = str(r["_id"])
        for k in ("start_date", "end_date"):
            try:
                r[k] = pd.to_datetime(r.get(k)).date()
            except Exception:
                r[k] = None
    df_i = pd.DataFrame(its).drop(columns=["_id"])
    return df_u.merge(df_i, on="itinerary_id", how="left")

def fetch_latest_followup_log_map(itinerary_ids: List[str]) -> Dict[str, dict]:
    if not itinerary_ids:
        return {}
    cur = col_followups.find({"itinerary_id": {"$in": itinerary_ids}})
    latest: Dict[str, dict] = {}
    for d in cur:
        iid = str(d.get("itinerary_id"))
        ts = _clean_dt(d.get("created_at")) or datetime.min
        if iid not in latest or ts > latest[iid].get("_ts", datetime.min):
            d["_ts"] = ts
            latest[iid] = d
    return latest

def fetch_confirmed_incentives(user: str, start_d: date, end_d: date) -> int:
    q = {
        "status": "confirmed",
        "rep_name": user,
        "booking_date": {
            "$gte": datetime.combine(start_d, datetime.min.time()),
            "$lte": datetime.combine(end_d, datetime.max.time())
        }
    }
    cur = col_updates.find(q, {"_id": 0, "incentive": 1})
    return sum(_to_int(d.get("incentive", 0)) for d in cur)

def count_confirmed(user: str, start_d: Optional[date]=None, end_d: Optional[date]=None) -> int:
    q = {"status": "confirmed", "rep_name": user}
    if start_d and end_d:
        q["booking_date"] = {
            "$gte": datetime.combine(start_d, datetime.min.time()),
            "$lte": datetime.combine(end_d, datetime.max.time())
        }
    return col_updates.count_documents(q)

# ----------------------------
# Updaters
# ----------------------------
def upsert_update_status(
    iid: str,
    status: str,
    user: str,
    next_followup_on: Optional[date],
    booking_date: Optional[date],
    comment: str,
    cancellation_reason: Optional[str],
    advance_amount: Optional[int],
) -> None:
    """
    Write a log into col_followups AND update col_updates latest status.
    Also, when confirming, stamp rep_name and incentive for the *current user*.
    """
    # 1) Immutable log
    log_doc = {
        "itinerary_id": str(iid),
        "created_at": datetime.utcnow(),
        "created_by": user,
        "status": status,  # "followup" | "confirmed" | "cancelled"
        "comment": str(comment or ""),
        "next_followup_on": (
            datetime.combine(next_followup_on, datetime.min.time())
            if next_followup_on else None
        ),
        "cancellation_reason": (str(cancellation_reason or "") if status == "cancelled" else ""),
    }
    base = col_itineraries.find_one({"_id": ObjectId(iid)}, {"client_name":1,"client_mobile":1,"ach_id":1})
    if base:
        log_doc.update({
            "client_name": base.get("client_name", ""),
            "client_mobile": base.get("client_mobile", ""),
            "ach_id": base.get("ach_id", ""),
        })
    col_followups.insert_one(log_doc)

    # 2) Latest status snapshot (+ incentive on confirm)
    upd = {
        "itinerary_id": str(iid),
        "status": status if status in ("followup", "cancelled") else "confirmed",
        "assigned_to": user if status == "followup" else None,
        "updated_at": datetime.utcnow(),
    }
    if status == "confirmed":
        # set booking info
        if booking_date:
            upd["booking_date"] = datetime.combine(booking_date, datetime.min.time())
        if advance_amount is not None:
            upd["advance_amount"] = int(advance_amount)
        # incentive & owner (this user)
        pkg_amt = _package_amount_for(iid)
        upd["incentive"] = _compute_incentive(pkg_amt)
        upd["rep_name"] = user  # attribute incentive to the user who confirmed
    if status == "cancelled":
        upd["cancellation_reason"] = str(cancellation_reason or "")

    col_updates.update_one({"itinerary_id": str(iid)}, {"$set": upd}, upsert=True)

def save_final_package_cost(iid: str, amount: int, user: str) -> None:
    """
    Persist final package cost to `expenses` and, if already confirmed,
    recompute the incentive in `package_updates`.
    """
    doc = {
        "itinerary_id": str(iid),
        "package_cost": int(amount),
        "saved_at": datetime.utcnow(),
    }
    col_expenses.update_one({"itinerary_id": str(iid)}, {"$set": doc}, upsert=True)

    # If confirmed, refresh incentive immediately based on new cost.
    upd = col_updates.find_one({"itinerary_id": str(iid)}, {"status": 1, "rep_name": 1})
    if upd and upd.get("status") == "confirmed":
        inc = _compute_incentive(int(amount))
        rep = upd.get("rep_name") or user
        col_updates.update_one(
            {"itinerary_id": str(iid)},
            {"$set": {"incentive": inc, "rep_name": rep}}
        )

# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="Follow-up Tracker", layout="wide")
st.title("📞 Follow-up Tracker")

user = _login()
if not user:
    st.stop()

# Summary header
df_assigned = fetch_assigned_followups(user)
itinerary_ids = df_assigned["itinerary_id"].astype(str).tolist()
latest_map = fetch_latest_followup_log_map(itinerary_ids)

# derive "next follow-up" & last comment from latest log
df_assigned["next_followup_on"] = df_assigned["itinerary_id"].map(
    lambda x: (latest_map.get(str(x), {}) or {}).get("next_followup_on")
)
df_assigned["next_followup_on"] = df_assigned["next_followup_on"].apply(
    lambda x: pd.to_datetime(x).date() if pd.notna(x) else None
)
df_assigned["last_comment"] = df_assigned["itinerary_id"].map(
    lambda x: (latest_map.get(str(x), {}) or {}).get("comment", "")
)

today = _today()
tmr = today + timedelta(days=1)
in7 = today + timedelta(days=7)

total_my_pkgs = len(df_assigned)
due_today = int((df_assigned["next_followup_on"] == today).sum())
due_tomorrow = int((df_assigned["next_followup_on"] == tmr).sum())
due_week = int(((df_assigned["next_followup_on"] >= today) &
                (df_assigned["next_followup_on"] <= in7)).sum())

# incentives & confirmation counts
first_this, last_this = month_bounds(today)
first_last, last_last = month_bounds(first_this - timedelta(days=1))
this_month_incentive = fetch_confirmed_incentives(user, first_this, last_this)
last_month_incentive = fetch_confirmed_incentives(user, first_last, last_last)
confirmed_this_month = count_confirmed(user, first_this, last_this)
confirmed_all_time   = count_confirmed(user)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Assigned follow-ups", total_my_pkgs)
c2.metric("Due today", due_today)
c3.metric("Due tomorrow", due_tomorrow)
c4.metric("Due in next 7 days", due_week)
c5.metric("My incentive", f"₹ {this_month_incentive:,}", help=f"Last month: ₹ {last_month_incentive:,}")

c6, c7 = st.columns(2)
c6.metric("Confirmed this month", confirmed_this_month)
c7.metric("Confirmed (all time)", confirmed_all_time)

st.divider()

# Table of assigned clients
st.subheader("My follow-ups")
if df_assigned.empty:
    st.info("No follow-ups assigned to you right now.")
    st.stop()

table = df_assigned[[
    "ach_id","client_name","client_mobile","start_date","end_date",
    "final_route","next_followup_on","last_comment","itinerary_id"
]].copy().sort_values(["next_followup_on","start_date"], na_position="last")
table.rename(columns={
    "ach_id":"ACH ID",
    "client_name":"Client",
    "client_mobile":"Mobile",
    "start_date":"Start",
    "end_date":"End",
    "final_route":"Route",
    "next_followup_on":"Next follow-up",
    "last_comment":"Last comment",
}, inplace=True)

left, right = st.columns([2, 1])
with left:
    st.dataframe(table.drop(columns=["itinerary_id"]), use_container_width=True, hide_index=True)
with right:
    options = (table["ACH ID"].fillna("").astype(str) + " | " +
               table["Client"].fillna("") + " | " +
               table["Mobile"].fillna("") + " | " +
               table["itinerary_id"])
    sel = st.selectbox("Open client", options.tolist())
    chosen_id = sel.split(" | ")[-1] if sel else None

if not chosen_id:
    st.stop()

st.divider()
st.subheader("Details & Update")

it_doc = col_itineraries.find_one({"_id": ObjectId(chosen_id)}) or {}
upd_doc = col_updates.find_one({"itinerary_id": str(chosen_id)}, {"_id": 0}) or {}

dc1, dc2 = st.columns([1, 1])
with dc1:
    st.markdown("**Client & Package**")
    st.write({
        "ACH ID": it_doc.get("ach_id",""),
        "Client": it_doc.get("client_name",""),
        "Mobile": it_doc.get("client_mobile",""),
        "Route": it_doc.get("final_route",""),
        "Pax": it_doc.get("total_pax",""),
        "Travel": f"{it_doc.get('start_date','')} → {it_doc.get('end_date','')}",
        "Representative": it_doc.get("representative",""),
    })
with dc2:
    st.markdown("**Current Status**")
    st.write({
        "Status": upd_doc.get("status",""),
        "Assigned To": upd_doc.get("assigned_to",""),
        "Booking date": upd_doc.get("booking_date",""),
        "Advance (₹)": upd_doc.get("advance_amount",0),
        "Incentive (₹)": upd_doc.get("incentive",0),
        "Rep (credited to)": upd_doc.get("rep_name",""),
    })

# ---- Final Package Cost editor ----
st.markdown("### Final package cost")
_current_cost = _package_amount_for(chosen_id)
new_cost = st.number_input(
    "Final package cost (₹)",
    min_value=0,
    step=500,
    value=int(_current_cost),
    help="This will be saved to Expenses as the final package cost. "
         "If the package is already confirmed, the incentive will be recalculated."
)
if st.button("💾 Save as final package cost"):
    try:
        save_final_package_cost(chosen_id, int(new_cost), user)
        st.success("Final package cost saved. Incentive updated if the package was confirmed.")
        st.rerun()
    except Exception as e:
        st.error(f"Could not save package cost: {e}")

with st.expander("Show full itinerary text"):
    st.text_area("Itinerary shared with client",
                 value=it_doc.get("itinerary_text",""), height=260, disabled=True)

# Follow-up trail
st.markdown("### Follow-up trail")
trail = list(col_followups.find({"itinerary_id": str(chosen_id)}).sort("created_at", -1))
if trail:
    df_trail = pd.DataFrame([{
        "When": t.get("created_at"),
        "By": t.get("created_by"),
        "Status": t.get("status"),
        "Next follow-up": (
            pd.to_datetime(t.get("next_followup_on")).date()
            if t.get("next_followup_on") else None
        ),
        "Comment": t.get("comment",""),
        "Cancel reason": t.get("cancellation_reason",""),
    } for t in trail])
    st.dataframe(df_trail, use_container_width=True, hide_index=True)
else:
    st.caption("No follow-up logs yet for this client.")

st.markdown("---")
st.markdown("### Add follow-up update")

with st.form("followup_form"):
    status = st.selectbox("Status", ["followup required", "confirmed", "cancelled"])
    comment = st.text_area("Comment", placeholder="Write your update…")

    next_date = None
    cancel_reason = None
    booking_date = None
    advance_amt = None

    if status == "followup required":
        next_date = st.date_input("Next follow-up on")
    elif status == "confirmed":
        booking_date = st.date_input("Booking date")
        advance_amt = st.number_input("Advance amount (₹) — optional", min_value=0, step=500, value=0)
    elif status == "cancelled":
        cancel_reason = st.text_input("Reason for cancellation", placeholder="Required")

    submitted = st.form_submit_button("💾 Save update")

if submitted:
    if status == "followup required" and not next_date:
        st.error("Please choose the next follow-up date."); st.stop()
    if status == "cancelled" and not (cancel_reason or "").strip():
        st.error("Please provide a reason for cancellation."); st.stop()
    if status == "confirmed" and not booking_date:
        st.error("Please choose the booking date."); st.stop()

    upsert_update_status(
        iid=chosen_id,
        status=("followup" if status == "followup required" else status),
        user=user,
        next_followup_on=next_date,
        booking_date=booking_date,
        comment=comment,
        cancellation_reason=cancel_reason,
        advance_amount=int(advance_amt) if advance_amt is not None else None
    )
    st.success("Update saved.")
    st.rerun()
