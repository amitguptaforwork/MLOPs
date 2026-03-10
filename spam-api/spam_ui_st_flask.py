import streamlit as st
import requests

# URL of your Flask API
API_URL = "http://127.0.0.1:5000/predict"

st.title("📧 Spam Detection Frontend")

message = st.text_area("Enter your message")

if st.button("Predict"):

    if message.strip() == "":
        st.warning("Please enter a message.")
    else:
        try:
            response = requests.post(
                API_URL,
                json={"message": message}
            )

            if response.status_code == 200:
                result = response.json()["prediction"]

                if result == "spam":
                    st.error("🚨 Spam detected!")
                else:
                    st.success("✅ Ham (Not Spam)")
            else:
                st.error("API Error: " + response.text)

        except requests.exceptions.ConnectionError:
            st.error("⚠️ Could not connect to Flask API. Make sure it is running.")