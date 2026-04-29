import streamlit as st
import requests

st.set_page_config(page_title="INVESTRA", layout="centered")
st.title("INVESTRA - Gold Decision Support")

current_price = st.number_input("Current price", min_value=0.0, value=2350.0)
predicted_price = st.number_input("Predicted price (t+3)", min_value=0.0, value=2365.0)
interval_low = st.number_input("Lower interval", min_value=0.0, value=2348.0)
interval_high = st.number_input("Upper interval", min_value=0.0, value=2378.0)

if st.button("Get Recommendation"):
    payload = {
        "current_price": current_price,
        "predicted_price": predicted_price,
        "interval_low": interval_low,
        "interval_high": interval_high,
    }
    try:
        response = requests.post("http://127.0.0.1:8000/recommend", json=payload, timeout=10)
        response.raise_for_status()
        result = response.json()
        st.success(f"Signal: {result['signal']}")
        st.write(result)
    except Exception as exc:
        st.error(f"API error: {exc}")
