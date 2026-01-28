import streamlit as st
import requests

st.set_page_config(page_title="Financial News AI", layout="wide")

st.title("🗞️ Stock News RAG Assistant")
st.markdown("Ask questions about **Apple** or **DeepSeek** based on your JSON news data.")

# Sidebar for configuration
with st.sidebar:
    st.header("Settings")
    api_url = st.text_input("API URL", value="http://127.0.0.1:8000/ask")

# User Input
query = st.text_input("Enter your question:", placeholder="e.g., What happened with Apple's modem?")

if st.button("Get Answer"):
    if query:
        with st.spinner("Searching news and generating answer..."):
            # Call the FastAPI backend
            try:
                payload = {"question": query}
                response = requests.post(api_url, json=payload)
                data = response.json()

                # Display Results in two columns
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.subheader("AI Analysis")
                    st.success(data["answer"])

                with col2:
                    st.subheader("Source Snippets")
                    for i, text in enumerate(data["sources"]):
                        with st.expander(f"Source {i+1}"):
                            st.write(text)
            except Exception as e:
                st.error(f"Could not connect to API: {e}")