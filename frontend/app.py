import streamlit as st
import requests

API_URL = "http://localhost:8000"  # Changed from "backend" to "localhost"

st.title("AI Text Summarizer")

text_input = st.text_area("Enter your text here:", height=200)
length_option = st.selectbox("Select summary length:", ["short", "medium", "long"])
method_option = st.selectbox("Select summarization method:", ["openai", "google-pegasus", "meta-bart"])

if st.button("Summarize"):
    if text_input:
        response = requests.post(f"{API_URL}/summarize", json={
            "text": text_input,
            "length": length_option,
            "method": method_option
        })
        if response.status_code == 200:
            st.subheader("Summary:")
            st.write(response.json()["summary"])
        else:
            st.error("An error occurred during summarization.")
    else:
        st.warning("Please enter some text to summarize.")

st.subheader("Recent Summaries")
history_response = requests.get(f"{API_URL}/history")
if history_response.status_code == 200:
    history = history_response.json()["history"]
    for item in history:

        st.text(f"Summary: {item['summary']}")
        st.text(f"Length: {item['length']}")
        st.text(f"Method: {item['method']}")
        st.text(f"Timestamp: {item['timestamp']}")
        st.text("---"*10)
else:
    st.error("Failed to fetch summary history.")
