import streamlit as st
import numpy as np
import pickle

# Load model
with open("f1_model.pkl", "rb") as f:
    bundle = pickle.load(f)

model = bundle["model"]
le_driver = bundle["le_driver"]
le_team = bundle["le_team"]
le_track = bundle["le_track"]
le_weather = bundle["le_weather"]

st.set_page_config(page_title="🏁 F1 Podium Predictor", layout="centered")
st.title("🏎️ Formula 1 Podium Predictor")
st.write("Enter race details to predict if the driver will finish on the podium.")

# Inputs
driver = st.selectbox("Driver", le_driver.classes_)
team = st.selectbox("Team", le_team.classes_)
qual_pos = st.slider("Qualifying Position", 1, 20, 5)
track = st.selectbox("Track", le_track.classes_)
weather = st.selectbox("Weather", le_weather.classes_)

# Encoding
driver_enc = le_driver.transform([driver])[0]
team_enc = le_team.transform([team])[0]
track_enc = le_track.transform([track])[0]
weather_enc = le_weather.transform([weather])[0]

input_data = np.array([[driver_enc, team_enc, qual_pos, track_enc, weather_enc]])

# Predict
if st.button("Predict 🏁"):
    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    if pred == 1:
        st.success(f"🎉 {driver} is likely to finish on the podium! ({prob*100:.1f}% confidence)")
    else:
        st.warning(f"❌ {driver} may not reach the podium. ({prob*100:.1f}% confidence)")

st.markdown("---")
st.caption("🛠️ This is a basic F1 demo model. Real data coming soon!")
