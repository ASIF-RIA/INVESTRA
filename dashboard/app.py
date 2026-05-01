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

st.divider()
st.subheader("Auto Predict (from trained model)")

if st.button("Get Latest Prediction"):
    try:
        response = requests.get("http://127.0.0.1:8000/predict", timeout=20)
        response.raise_for_status()
        result = response.json()
        st.success(f"Predicted price: {result['predicted_price']:.4f}")
        st.write(result)
        st.info("Use this predicted value in the recommendation section above.")
    except Exception as exc:
        st.error(f"Prediction API error: {exc}")

st.divider()
st.subheader("Automatic Daily Forecast")

try:
    forecast_resp = requests.get("http://127.0.0.1:8000/forecast/latest", timeout=30)
    forecast_resp.raise_for_status()
    forecast = forecast_resp.json()

    tomorrow = forecast["tomorrow_prediction"]
    direction = tomorrow["direction_vs_today"].upper()
    st.success(
        f"Tomorrow ({tomorrow['date']}): {tomorrow['predicted_close']:.4f} | Expected to {direction}"
    )
    st.caption(f"Forecast generated on: {forecast['generated_on']} | Model: {forecast['model']}")

    st.write("Next 7-day predicted close prices:")
    st.dataframe(forecast["next_week_predictions"], use_container_width=True)
except Exception as exc:
    st.warning(f"Automatic daily forecast is unavailable right now: {exc}")
