# frontend/streamlit_app.py

import streamlit as st
import requests

# FastAPI Backend URL
API_URL = "http://127.0.0.1:8000"

st.title("📄 AI-Powered RFP Automation System")

extracted_rfp_text = ""

# 📂 Upload RFP Document
st.header("📂 Upload an RFP Document")
uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])

if uploaded_file:
    files = {"file": uploaded_file.getvalue()}
    response = requests.post(f"{API_URL}/rfp/upload_rfp", files=files)

    if response.status_code == 200:
        result = response.json()
        extracted_rfp_text = result["extracted_text"]
        st.success(f"✅ File Uploaded: {result['filename']}")
        st.write("📜 **Extracted Text Preview:**", extracted_rfp_text[:500])

        retrieval_response = requests.get(f"{API_URL}/retrieval/retrieve_docs?query={extracted_rfp_text}")

        if retrieval_response.status_code == 200:
            retrieved_docs = retrieval_response.json().get("retrieved_docs", [])
            response = requests.post(f"{API_URL}/proposal/generate_proposal", 
                                     json={"rfp_text": extracted_rfp_text, "retrieved_docs": retrieved_docs})
            
            if response.status_code == 200:
                result = response.json()
                st.success("✅ Proposal Generated!")
                st.write("📌 **Generated Proposal:**", result["proposal"])
                requests.post(f"{API_URL}/proposal/store_proposal", json={"proposal": result["proposal"]})
            else:
                st.error(f"❌ Error generating proposal: {response.text}")
        else:
            st.error(f"❌ Error retrieving documents: {retrieval_response.text}")
    else:
        st.error(f"❌ Error uploading file: {response.text}")

st.header("🛠️ Refine Proposal")
user_feedback = st.text_area("Your Feedback", height=100)

if st.button("Refine Proposal", key="refine_proposal"):
    if user_feedback:
        refine_response = requests.post(f"{API_URL}/proposal/refine_proposal", json={"user_feedback": user_feedback})

        if refine_response.status_code == 200:
            result = refine_response.json()
            refined_proposal = result.get("refined_proposal", "")

            if refined_proposal:
                st.success("✅ Proposal Refined!")
                st.write("📌 **Refined Proposal:**", refined_proposal)

                # ✅ Update the stored proposal for the UI
                requests.post(f"{API_URL}/proposal/store_proposal", json={"proposal": refined_proposal})
            else:
                st.warning("⚠️ No changes were made.")
        else:
            st.error(f"❌ Error refining proposal: {refine_response.text}")
    else:
        st.warning("⚠️ Please enter feedback to refine the proposal.")