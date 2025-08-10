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


import streamlit as st
import pandas as pd
import io
import requests
import math
import locale
import datetime
from pymongo import MongoClient

# -----------------------------
# Static files on GitHub
# -----------------------------
CODE_FILE_URL = "https://raw.githubusercontent.com/Arpith92/TAK-Project/main/Code.xlsx"
BHASMARATHI_TYPE_URL = "https://raw.githubusercontent.com/Arpith92/TAK-Project/main/Bhasmarathi_Type.xlsx"
STAY_CITY_URL = "https://raw.githubusercontent.com/Arpith92/TAK-Project/main/Stay_City.xlsx"

# -----------------------------
# Helpers
# -----------------------------
def read_excel_from_url(url, sheet_name=None):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return pd.read_excel(io.BytesIO(response.content), sheet_name=sheet_name)
    except Exception as e:
        st.error(f"Error reading file from {url}: {e}")
        return None

def is_valid_mobile(num: str) -> bool:
    if num is None:
        return False
    digits = "".join(ch for ch in str(num) if ch.isdigit())
    return len(digits) == 10

# -----------------------------
# UI
# -----------------------------
st.title("TAK Project Itinerary Generator")

uploaded_file = st.file_uploader("Upload date-based Excel file", type=["xlsx"])

# Only show client dropdown *after* file is uploaded
if not uploaded_file:
    st.info("⬆️ Upload the Excel to continue.")
    st.stop()

# Read workbook to get sheet names for dropdown
try:
    input_data = pd.ExcelFile(uploaded_file)
except Exception as e:
    st.error(f"Error reading uploaded file: {e}")
    st.stop()

sheet_options = input_data.sheet_names or []
if not sheet_options:
    st.error("No sheets found in the uploaded Excel.")
    st.stop()

client_name = st.selectbox("Select client (sheet name)", sheet_options, index=0)

# Mobile + Representative
client_mobile_raw = st.text_input("Enter client mobile number (10 digits)").strip()
rep_options = ["-- Select --", "Arpith", "Reena", "Kuldeep", "Teena"]
representative = st.selectbox("Representative name", rep_options)

# Guard: require valid inputs
if not client_name or not client_mobile_raw or representative == "-- Select --":
    st.info("Enter valid mobile and choose a representative to proceed.")
    st.stop()

if not is_valid_mobile(client_mobile_raw):
    st.error("❌ Invalid mobile number. Please enter a 10-digit number (digits only).")
    st.stop()

client_mobile = "".join(ch for ch in client_mobile_raw if ch.isdigit())

# -----------------------------
# Parse selected sheet
# -----------------------------
try:
    client_data = input_data.parse(sheet_name=client_name)
except Exception as e:
    st.error(f"Error reading selected sheet: {e}")
    st.stop()

st.success(f"'{client_name}' sheet selected. Proceeding with processing...")

# ---- Validate travel dates are not in the past ----
if "Date" not in client_data.columns:
    st.error("❌ 'Date' column is missing in the uploaded sheet.")
    st.stop()

dates_series = pd.to_datetime(client_data["Date"], errors="coerce").dt.date
if dates_series.isna().all():
    st.error("❌ No valid dates found in the uploaded file.")
    st.stop()

#today = datetime.date.today()
#if (dates_series < today).any():
#    st.error("❌ Travel dates are in the past.")
#    st.stop()

# -----------------------------
# Load static Excel files
# -----------------------------
stay_city_df = read_excel_from_url(STAY_CITY_URL, sheet_name="Stay_City")
code_df = read_excel_from_url(CODE_FILE_URL, sheet_name="Code")
bhasmarathi_type_df = read_excel_from_url(BHASMARATHI_TYPE_URL, sheet_name="Bhasmarathi_Type")

# -----------------------------
# Build itinerary from codes
# -----------------------------
itinerary = []
for _, row in client_data.iterrows():
    code = row.get('Code', None)
    if code is None:
        itinerary.append({
            'Date': row.get('Date', 'N/A'),
            'Time': row.get('Time', 'N/A'),
            'Description': "No code provided in row"
        })
        continue

    particulars = code_df.loc[code_df['Code'] == code, 'Particulars'].values if code_df is not None else []
    description = particulars[0] if len(particulars) > 0 else f"No description found for code {code}"

    itinerary.append({
        'Date': row.get('Date', 'N/A'),
        'Time': row.get('Time', 'N/A'),
        'Description': description
    })

# -----------------------------
# Totals & route
# -----------------------------
start_date = pd.to_datetime(client_data['Date'].min())
end_date = pd.to_datetime(client_data['Date'].max())
total_days = (end_date - start_date).days + 1
total_nights = total_days - 1

if "Total Pax" not in client_data.columns:
    st.error("❌ 'Total Pax' column is missing in the uploaded sheet.")
    st.stop()
total_pax = int(client_data['Total Pax'].iloc[0])

night_text = "Night" if total_nights == 1 else "Nights"
person_text = "Person" if total_pax == 1 else "Persons"

# Route from codes
route_parts = []
if code_df is not None and 'Route' in code_df.columns and 'Code' in client_data.columns:
    for code in client_data['Code']:
        matched_routes = code_df.loc[code_df['Code'] == code, 'Route']
        if not matched_routes.empty:
            route_parts.append(matched_routes.iloc[0])

route = '-'.join(route_parts).replace(' -', '-').replace('- ', '-')
route_list = route.split('-') if route else []
final_route = '-'.join([route_list[i] for i in range(len(route_list)) if i == 0 or route_list[i] != route_list[i - 1]]) if route_list else ""

# Package cost calc
def calculate_package_cost(df):
    for required in ['Car Cost', 'Hotel Cost', 'Bhasmarathi Cost']:
        if required not in df.columns:
            df[required] = 0
    car_cost = df['Car Cost'].sum()
    hotel_cost = df['Hotel Cost'].sum()
    bhasmarathi_cost = df['Bhasmarathi Cost'].sum()
    total = car_cost + hotel_cost + bhasmarathi_cost
    return math.ceil(total / 1000) * 1000 - 1 if total > 0 else 0

try:
    locale.setlocale(locale.LC_ALL, 'en_IN')
    use_locale = True
except locale.Error:
    use_locale = False

total_package_cost = calculate_package_cost(client_data)
formatted_cost = (
    locale.format_string("%d", total_package_cost, grouping=True)
    if use_locale else f"{total_package_cost:,}"
)
formatted_cost1 = formatted_cost.replace(",", "X").replace("X", ",", 1)

# Types & details line
car_types = client_data['Car Type'].dropna().unique() if 'Car Type' in client_data.columns else []
car_types_str = '-'.join(car_types) if len(car_types) else ""

hotel_types = client_data['Hotel Type'].dropna().unique() if 'Hotel Type' in client_data.columns else []
hotel_types_str = '-'.join(hotel_types) if len(hotel_types) else ""

bhasmarathi_types = client_data['Bhasmarathi Type'].dropna().unique() if 'Bhasmarathi Type' in client_data.columns else []
bhasmarathi_descriptions = []
if bhasmarathi_type_df is not None and 'Bhasmarathi Type' in bhasmarathi_type_df.columns and 'Description' in bhasmarathi_type_df.columns:
    for bhas_type in bhasmarathi_types:
        match = bhasmarathi_type_df.loc[bhasmarathi_type_df['Bhasmarathi Type'] == bhas_type, 'Description']
        if not match.empty:
            bhasmarathi_descriptions.append(match.iloc[0])
bhasmarathi_desc_str = '-'.join(bhasmarathi_descriptions)
details_line = f"({car_types_str},{hotel_types_str},{bhasmarathi_desc_str})".strip("(),")

# -----------------------------
# Build itinerary text (NO mobile/rep lines)
# -----------------------------
greeting = f"Greetings from TravelAajkal,\n\n*Client Name: {client_name}*\n\n"
plan = f"*Plan:- {total_days}Days and {total_nights}{night_text} {final_route} for {total_pax} {person_text}*"

itinerary_message = greeting + plan + "\n\n*Itinerary:*\n"
grouped_itinerary = {}

for entry in itinerary:
    if entry['Date'] != 'N/A' and pd.notna(entry['Date']):
        date_str = pd.to_datetime(entry['Date']).strftime('%d-%b-%Y')
        if date_str not in grouped_itinerary:
            grouped_itinerary[date_str] = []
        grouped_itinerary[date_str].append(f"{entry['Time']}: {entry['Description']}")

day_number = 1
first_day = True
for date_str, events in grouped_itinerary.items():
    itinerary_message += f"\n*Day{day_number}:{date_str}*\n"
    for event in events:
        itinerary_message += f"{event if first_day else event[5:]}\n"
        first_day = False
    day_number += 1

itinerary_message += f"\n*Package cost: {formatted_cost1}/-*\n{details_line}"

# -----------------------------
# Inclusions
# -----------------------------
inclusions = []
if car_types_str:
    inclusions.append(f"Entire travel as per itinerary by {car_types_str}.")
    inclusions.append("Toll, parking, and driver bata are included.")
    inclusions.append("Airport/ Railway station pickup and drop.")

if bhasmarathi_desc_str:
    inclusions.append(f"{bhasmarathi_desc_str} for {total_pax} {person_text}.")
    inclusions.append("Bhasm-Aarti pickup and drop.")

if "Stay City" in client_data.columns and "Room Type" in client_data.columns and stay_city_df is not None:
    city_nights = {}
    for i in range(len(client_data)):
        stay_city = client_data["Stay City"].iloc[i]
        if pd.isna(stay_city):
            continue
        stay_city = str(stay_city).strip()
        if i > 0 and client_data["Stay City"].iloc[i] == client_data["Stay City"].iloc[i - 1]:
            city_nights[stay_city] = city_nights.get(stay_city, 1) + 1
        else:
            city_nights[stay_city] = 1

    total_used_nights = 0
    for i in range(len(client_data)):
        stay_city = client_data["Stay City"].iloc[i]
        room_type = client_data["Room Type"].iloc[i]
        if pd.isna(stay_city):
            continue
        stay_city = str(stay_city).strip()

        matching_row = stay_city_df[stay_city_df["Stay City"] == stay_city]
        if not matching_row.empty:
            city_name = matching_row["City"].iloc[0]
            nights_here = city_nights.get(stay_city, 0)
            if total_used_nights + nights_here <= total_nights:
                inclusions.append(f"{nights_here}Night stay in {city_name} with {room_type} in {hotel_types_str}.")
                total_used_nights += nights_here
            else:
                break

if hotel_types_str:
    inclusions.append("*Standard check-in at 12:00 PM and check-out at 09:00 AM.*")
    inclusions.append("Early check-in and late check-out are subject to room availability.")

inclusions_section = "*Inclusions:-*\n" + "\n".join([f"{i + 1}. {line}" for i, line in enumerate(inclusions)])
final_message = itinerary_message + "\n\n" + inclusions_section

# -----------------------------
# Exclusions & Important Notes
# -----------------------------
exclusions = []
exclusions.append("Any meals or beverages not specified in the itinerary are not included. (e.g., Breakfast, lunch, dinner, snacks, personal beverages).")
if car_types_str:
    exclusions.append("Entry fees for any tourist attractions, temples, or monuments not specified in the inclusions.")
exclusions.append("Travel insurance.")
if car_types_str:
    exclusions.append("Expenses related to personal shopping, tips, or gratuities.")
if hotel_types_str:
    exclusions.append("Any additional charges for early check-in or late check-out if rooms are not available.")
if car_types_str:
    exclusions.append("Costs arising due to natural events, unforeseen roadblocks, or personal travel changes.")
if car_types_str:
    exclusions.append("Additional charges for any sightseeing spots not listed in the itinerary.")
exclusions_section = "*Exclusions:-*\n" + "\n".join([f"{i + 1}. {line}" for i, line in enumerate(exclusions)])

important_notes = []
if car_types_str:
    important_notes.append("Any tourist attractions not mentioned in the itinerary will incur additional charges.")
if car_types_str:
    important_notes.append("Visits to tourist spots or temples are subject to traffic conditions and temple management restrictions. If any tourist spot or temple is closed on the specific day of travel due to unforeseen circumstances, TravelaajKal will not be responsible, and no refunds will be provided.")
if bhasmarathi_desc_str:
    important_notes.append("For Bhasm-Aarti, we will provide tickets, but timely arrival at the temple and seating arrangements are beyond our control.")
if bhasmarathi_desc_str:
    important_notes.append("We only facilitate the booking of Bhasm-Aarti tickets. The ticket cost will be charged at actuals, as mentioned on the ticket.")
if bhasmarathi_desc_str:
    important_notes.append("No commitment can be made regarding ticket availability. Bhasm-Aarti tickets are subject to availability and may be canceled at any time based on the decisions of the temple management committee. In case of an unconfirmed ticket, the ticket cost will be refunded.")
if hotel_types_str:
    important_notes.append("Entry to the hotel is subject to the hotel's rules and regulations. A valid ID proof (Indian citizenship) is required. Only married couples are allowed entry.")
if hotel_types_str:
    important_notes.append("Children above 9 years will be considered as adults. Children under 9 years must share the same bed with parents. If an extra bed is required, additional charges will apply.")
important_notes_section = "\n*Important Notes:-*\n" + "\n".join([f"{i + 1}. {line}" for i, line in enumerate(important_notes)])

# -----------------------------
# Final output text
# -----------------------------
cancellation_policy = """
*Cancellation Policy:-*
1. 30+ days before travel → 20% of the advance amount will be deducted.
2. 15-29 days before travel → 50% of the advance amount will be deducted.
3. Less than 15 days before travel → No refund on the advance amount.
4. No refund for no-shows, last-minute cancellations, or early departures.
5. One-time rescheduling is allowed if requested at least 15 days before the travel date, subject to availability.
"""

payment_terms = """*Payment Terms:-*
50% advance and remaining 50% after arrival at Ujjain.
"""

booking_confirmation = """For booking confirmation, please make the advance payment to the company's current account provided below.

*Company Account details:-*
Account Name: ACHALA HOLIDAYS PVT LTD
Bank: Axis Bank
Account No: 923020071937652
IFSC Code: UTIB0000329
MICR Code: 452211003
Branch Address: Ground Floor, 77, Dewas Road, Ujjain, Madhya Pradesh 456010

Regards,
Team TravelAajKal™️
Reg. Achala Holidays Pvt Limited
Visit :- www.travelaajkal.com
Follow us :- https://www.instagram.com/travelaaj_kal/

*Great news! ACHALA HOLIDAYS PVT LTD is now a DPIIT-recognized Startup by the Government of India.*
*Thank you for your support as we continue to redefine travel.*
*Travel Aaj aur Kal with us!*

TravelAajKal® is a registered trademark of Achala Holidays Pvt Ltd.
"""

final_output = f"""
{final_message}

{exclusions_section}

{important_notes_section}

{cancellation_policy}

{payment_terms}

{booking_confirmation}
"""

st.subheader("Final Itinerary Details")
st.text_area("Preview", final_output, height=800)

st.download_button(
    label="📋 Copy / Download Itinerary",
    data=final_output,
    file_name="itinerary.txt",
    mime="text/plain"
)

# -----------------------------
# Save to MongoDB
# -----------------------------
MONGO_URI = "mongodb+srv://TAK_USER:Arpith%2692@cluster0.ewncl10.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
mongo_client = MongoClient(MONGO_URI)
db = mongo_client["TAK_DB"]
collection = db["itineraries"]

record = {
    "client_name": client_name,
    "client_mobile": client_mobile,           # NEW
    "representative": representative,         # NEW
    "upload_date": datetime.datetime.utcnow(),
    "start_date": str(start_date.date()),
    "end_date": str(end_date.date()),
    "total_days": total_days,
    "total_pax": total_pax,
    "final_route": final_route,
    "car_types": car_types_str,
    "hotel_types": hotel_types_str,
    "bhasmarathi_types": bhasmarathi_desc_str,
    "package_cost": formatted_cost1,
    "itinerary_text": final_output
}

try:
    collection.insert_one(record)
    st.success("✅ Itinerary saved to MongoDB")
except Exception as e:
    st.error(f"❌ Failed to save: {e}")
