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
        st.write("📊 **Complexity Score:**", result["complexity_score"])
    else:
        st.error(f"❌ Error uploading file: {response.text}")

# 📝 Generate Proposal
st.header("✍️ Generate Proposal from RFP Content")

rfp_text = st.text_area("Enter RFP content for proposal generation (optional)", height=200)

if st.button("Generate Proposal"):
    final_rfp_text = rfp_text.strip() if rfp_text.strip() else extracted_rfp_text.strip()

    if final_rfp_text:
        # ✅ First, retrieve similar documents from Pinecone
        retrieval_response = requests.get(f"{API_URL}/retrieval/retrieve_docs?query={final_rfp_text}")

        if retrieval_response.status_code == 200:
            retrieved_docs = retrieval_response.json().get("retrieved_docs", [])
            st.write("📌 **Retrieved Documents for Context:**")
            st.json(retrieved_docs)  # ✅ Display retrieved documents for debugging

            # ✅ Pass retrieved_docs along with the RFP text to proposal generation
            response = requests.post(f"{API_URL}/proposal/generate_proposal", 
                                     json={"rfp_text": final_rfp_text, "retrieved_docs": retrieved_docs})

            if response.status_code == 200:
                result = response.json()
                st.success("✅ Proposal Generated!")
                st.write("📌 **Generated Proposal:**", result["proposal"])
            else:
                st.error(f"❌ Error generating proposal: {response.text}")
        else:
            st.error(f"❌ Error retrieving documents: {retrieval_response.text}")
    else:
        st.warning("⚠️ Please upload an RFP file or enter text.")