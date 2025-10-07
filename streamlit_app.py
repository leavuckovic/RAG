import streamlit as st
import requests

st.title("📄 Document QA Chat")

# Session state to store session_id
if "session_id" not in st.session_state:
    st.session_state.session_id = None

# Upload documents
st.header("Upload Documents")
uploaded_files = st.file_uploader("Choose PDF or image files", type=["pdf", "png", "jpg", "jpeg"], accept_multiple_files=True)

if st.button("Upload"):
    if uploaded_files:
        files = [("documents", (f.name, f.read())) for f in uploaded_files]
        res = requests.post("http://backend:5000/upload", files=files)
        if res.ok:
            st.session_state.session_id = res.json()["session_id"]
            st.success(f"Uploaded! Session ID: {st.session_state.session_id}")
        else:
            st.error("Upload failed.")
    else:
        st.warning("Please select at least one file.")

# Ask a question
st.header("Ask a Question")
question = st.text_input("Type your question")

if st.button("Ask"):
    if not st.session_state.session_id:
        st.warning("Upload a document first.")
    elif question:
        payload = {
            "session_id": st.session_state.session_id,
            "question": question
        }
        res = requests.post("http://backend:5000/ask", json=payload)
        if res.ok:
            st.markdown(f"**Answer:** {res.json()['answer']}")
            st.markdown(f"**NER analysis:** {res.json()['entities']}")
        else:
            st.error("Failed to get answer.")