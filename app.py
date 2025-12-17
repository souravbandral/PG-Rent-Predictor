# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import io

# ---------------- page config ----------------
st.set_page_config(
    page_title="Chandigarh PG — Smart Rent",
    page_icon="🏠",
    layout="centered"
)

def set_dark_mode(enable: bool):
    if enable:
        dark_css = """
        <style>
        /* Entire background */
        .stApp {
            background-color: #121212 !important;
            color: #EAEAEA !important;
        }

        /* Headings */
        h1, h2, h3, h4 {
            color: #ffffff !important;
        }

        /* Text */
        .stMarkdown, .stText, label, span {
            color: #EAEAEA !important;
        }

        /* Input fields */
        .stSelectbox, .stNumberInput, textarea, input {
            background-color: #1E1E1E !important;
            color: #FFFFFF !important;
        }

        /* Dropdown arrow */
        .css-1gtu0rj {
            color: white !important;
        }

        /* Buttons */
        .stButton>button {
            background-color: #1E88E5 !important;
            color: white !important;
            border-radius: 8px !important;
            padding: 8px 16px !important;
        }

        /* Checkbox */
        .stCheckbox > label {
            color: #EAEAEA !important;
        }

        /* Sidebar */
        [data-testid="stSidebar"] {
            background-color: #1A1A1A !important;
            color: #EAEAEA !important;
        }

        </style>
        """
        st.markdown(dark_css, unsafe_allow_html=True)


# ---------------- utilities ----------------
def safe_load_joblib(path):
    try:
        return joblib.load(path)
    except Exception:
        return None

def prepare_row_for_features(inp_dict, feature_list):
    row = pd.DataFrame([inp_dict])
    for c in feature_list:
        if c not in row.columns:
            row[c] = 0
    return row[feature_list]

def df_to_csv_bytes(df):
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode('utf-8')

# ---------------- load models (if any) ----------------
rent_model = safe_load_joblib("rent_model.joblib")
rent_features = []
if rent_model is not None and os.path.exists("rent_model_features.txt"):
    try:
        rent_features = pd.read_csv("rent_model_features.txt", header=None)[0].tolist()
    except Exception:
        rent_features = []
have_rent_model = rent_model is not None and len(rent_features) > 0

price_model = safe_load_joblib("model.joblib")
price_features = []
if os.path.exists("model_features.txt"):
    try:
        price_features = pd.read_csv("model_features.txt", header=None)[0].tolist()
    except Exception:
        price_features = []
have_price_model = price_model is not None and len(price_features) > 0

# ---------------- static sector coordinates (for map) ----------------
# approximate lat/lon for demonstration (Chandigarh)
sector_coords = {
    "sector_15": (30.7300, 76.7800),
    "sector_17": (30.7350, 76.7900),
    "sector_21": (30.7450, 76.7800),
    "sector_22": (30.7375, 76.7920),
    "sector_25": (30.7330, 76.7770),
    "sector_27": (30.7290, 76.8080),
    "sector_34": (30.7050, 76.7680),
    "sector_35": (30.7075, 76.7520),
    "sector_38": (30.7220, 76.7440),
    "sector_41": (30.7180, 76.7600),
}

# ---------------- sidebar controls ----------------
st.sidebar.header("Controls & Theme")
dark_mode = st.sidebar.checkbox("Dark mode", value=False)
set_dark_mode(dark_mode)

show_map = st.sidebar.checkbox("Show sector map", value=True)
show_badges = st.sidebar.checkbox("Show emoji badges", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("Made for: **Chandigarh (student / PG use-case)**")
st.sidebar.markdown("Tip: add more real listings and re-train `rent_model` for production-quality results.")

# ---------------- header ----------------
st.title("🏠 Chandigarh PG — Smart Rent")
st.write("Predict monthly rent for PG / rooms. (Model-based if trained, else fallback + rule adjustments.)")

# ---------------- inputs ----------------
st.header("Room & Preference")
c1, c2 = st.columns(2)
with c1:
    area = st.number_input("Area (sqft)", value=120, min_value=20)
    bhk = st.selectbox("BHK", [1,2,3], index=0)
    furnished = st.selectbox("Furnished?", ["No","Yes"])
with c2:
    shared = st.selectbox("Shared room?", ["Yes","No"])
    food = st.selectbox("Food included?", ["No","Yes"])
    distance = st.number_input("Distance to college (km)", value=1.0, step=0.1)

# sector dropdown
sectors = list(sector_coords.keys()) + ["other"]
sector = st.selectbox("Sector / Locality", sectors, index=0)
if sector == "other":
    sector = st.text_input("Enter custom sector (exact model column name)", "sector_custom")

st.write("---")

# ---------------- amenities (feature #3) ----------------
st.subheader("Amenities (affect rent)")
amen_wifi = st.checkbox("WiFi (shared)", value=False)
amen_laundry = st.checkbox("Laundry", value=False)
amen_ac = st.checkbox("AC available", value=False)
amen_parking = st.checkbox("Parking", value=False)
amen_attached = st.checkbox("Attached bathroom", value=False)

# calculate small amenities multiplier (simple)
amen_mult = 0.0
if amen_wifi: amen_mult += 0.03    # +3%
if amen_laundry: amen_mult += 0.02  # +2%
if amen_ac: amen_mult += 0.08       # +8%
if amen_parking: amen_mult += 0.03  # +3%
if amen_attached: amen_mult += 0.07 # +7%

# ---------------- show map (feature #6) ----------------
if show_map:
    st.subheader("Sector Map (approx.)")
    try:
        # create small DataFrame for map
        map_rows = []
        for s, (lat, lon) in sector_coords.items():
            map_rows.append({"sector": s, "lat": lat, "lon": lon})
        map_df = pd.DataFrame(map_rows)
        # show with labels (st.map uses lat/lon)
        st.map(map_df.rename(columns={"lat":"lat","lon":"lon"}))
        # small legend
        st.caption("Map shows approximate sector locations for quick reference.")
    except Exception:
        st.info("Map cannot be displayed here.")

st.write("---")

# ---------------- predict logic ----------------
if st.button("Predict Rent"):

    # build base input dict
    inp = {
        "area_sqft": area,
        "bhk": bhk,
        "shared_room": 1 if shared=="Yes" else 0,
        "food_included": 1 if food=="Yes" else 0,
        "furnished": 1 if furnished=="Yes" else 0,
        "distance_km": distance
    }

    # model prediction attempt
    base_pred = None
    if have_rent_model:
        try:
            row = prepare_row_for_features(inp, rent_features)
            if sector in rent_features:
                row[sector] = 1
            row = row[rent_features]
            base_val = rent_model.predict(row)[0]
            base_pred = int(base_val)
        except Exception as e:
            st.error(f"Model prediction error: {e}")
            base_pred = None

    # fallback: try price_model -> rent multiplier
    if base_pred is None:
        if have_price_model:
            try:
                p_row = pd.DataFrame([inp])
                for c in price_features:
                    if c not in p_row.columns:
                        p_row[c] = 0
                if sector in price_features:
                    p_row[sector] = 1
                p_row = p_row[price_features]
                sale_pred = price_model.predict(p_row)[0]
                try:
                    sale_price = float(np.exp(sale_pred))
                except:
                    sale_price = float(sale_pred)
                base_pred = int(max(1200, sale_price * 0.0045))
                st.info("Using price->rent fallback (approx).")
            except Exception:
                base_pred = None
        else:
            base_pred = 4500
            st.info("No model found — using safe default base rent for demo.")

    # apply rule-based adjustments for BHK/food/furnished/shared
    bhk_factor = {1:1.0, 2:1.35, 3:1.75}.get(int(bhk), 1.0)
    food_mult = 0.10 if food=="Yes" else 0.0
    furn_mult = 0.08 if furnished=="Yes" else 0.0
    shared_mult = -0.30 if shared=="Yes" else 0.0

    # apply amenities multiplier
    total_mult = 1 + food_mult + furn_mult + amen_mult + shared_mult

    adjusted = int(max(300, round(base_pred * bhk_factor * total_mult)))

    # show results
    st.success(f"### Estimated monthly rent: ₹{adjusted:,}")
    st.write(f"Base model estimate: ₹{base_pred:,}   ·  BHK factor: ×{bhk_factor}")
    st.write(f"Amenities adjustment: {int((amen_mult)*100)}%  ·  Food: +{int(food_mult*100)}%  · Furnished: +{int(furn_mult*100)}%  · Shared: {int(shared_mult*100)}%")

    # per-person if shared
    if shared == "Yes":
        roommates = st.number_input("How many people sharing (including you)?", min_value=2, value=2)
        per_person = adjusted // max(1, roommates)
        st.info(f"Per-person rent ≈ ₹{per_person:,} /month")

    st.write("---")

    # ---------------- emoji badges (feature #10) ----------------
    if show_badges:
        badges = []
        if furnished=="Yes": badges.append("🛏️ Furnished")
        else: badges.append("🛋️ Unfurnished")
        if food=="Yes": badges.append("🍽️ Food included")
        else: badges.append("🍳 No food")
        if shared=="Yes": badges.append("👥 Shared room")
        else: badges.append("🔒 Private room")
        if amen_wifi: badges.append("📶 WiFi")
        if amen_laundry: badges.append("🧺 Laundry")
        if amen_ac: badges.append("❄️ AC")
        if amen_parking: badges.append("🚗 Parking")
        if amen_attached: badges.append("🚿 Attached bath")
        if distance <= 1.0:
            badges.append("📍 Near college")
        # show badges inline
        cols = st.columns(len(badges) if len(badges)>0 else 1)
        for i, b in enumerate(badges):
            with cols[i]:
                st.markdown(f"**{b}**")

    st.write("---")
    st.caption("Note: This is a demo-level estimator. For production accuracy, collect more listings and retrain the model.")
