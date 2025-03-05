import streamlit as st
import requests
from fpdf import FPDF
import time

# FastAPI Backend URL
API_URL = "http://127.0.0.1:8000"

st.title("📄 AI-Powered RFP Automation System")

# Initialize Session State Variables
if "current_proposal" not in st.session_state:
    st.session_state.current_proposal = ""
if "proposal_generated" not in st.session_state:
    st.session_state.proposal_generated = False
if "proposal_refined" not in st.session_state:
    st.session_state.proposal_refined = False

# --- Step 1: Upload and Generate Proposal (only once) ---
st.header("📂 Upload an RFP Document")
if not st.session_state.proposal_generated:
    uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])
    if uploaded_file:
        files = {"file": uploaded_file.getvalue()}
        response = requests.post(f"{API_URL}/rfp/upload_rfp", files=files)
        if response.status_code == 200:
            result = response.json()
            extracted_rfp_text = result["extracted_text"]
            st.success(f"✅ File Uploaded: {result['filename']}")
            st.write("📜 **Extracted Text Preview:**", extracted_rfp_text[:500])
            # Generate the initial proposal
            proposal_response = requests.post(
                f"{API_URL}/proposal/generate_proposal",
                json={"rfp_text": extracted_rfp_text, "retrieved_docs": []}
            )
            if proposal_response.status_code == 200:
                result = proposal_response.json()
                st.session_state.current_proposal = result["proposal"]
                st.success("✅ Proposal Generated!")
                st.write("📌 **Generated Proposal:**", result["proposal"])
                st.session_state.proposal_generated = True
                # Store the proposal in the backend
                requests.post(f"{API_URL}/proposal/store_proposal", json={"proposal": result["proposal"]})
            else:
                st.error(f"❌ Error generating proposal: {proposal_response.text}")
        else:
            st.error("❌ File upload failed.")
else:
    st.write("Using previously generated proposal:")
    st.write(st.session_state.current_proposal)

# --- Step 2: Refine the Proposal ---
st.header("🛠️ Refine Proposal")
user_feedback = st.text_area("Your Feedback", height=100)

if st.button("Refine Proposal"):
    if user_feedback:
        refine_response = requests.post(
            f"{API_URL}/proposal/refine_proposal",
            json={"user_feedback": user_feedback}
        )
        if refine_response.status_code == 200:
            result = refine_response.json()
            refined_proposal = result.get("refined_proposal", "")
            if refined_proposal:
                st.session_state.current_proposal = refined_proposal
                st.session_state.proposal_refined = True
                st.success("✅ Proposal Refined!")
                st.write("📌 **Refined Proposal:**", refined_proposal)
                # Update the backend with the refined proposal
                requests.post(f"{API_URL}/proposal/store_proposal", json={"proposal": refined_proposal})
            else:
                st.warning("⚠️ No changes were made.")
        else:
            st.error(f"❌ Error refining proposal: {refine_response.text}")

# --- Step 3: Export the Latest Proposal as PDF ---
st.header("📤 Finalize & Export")
if st.button("Submit and Export as PDF"):
    # Always fetch the latest proposal from the backend (with cache busting)
    response = requests.get(f"{API_URL}/proposal/get_latest_proposal?timestamp={time.time()}")
    if response.status_code == 200:
        final_proposal_text = response.json().get("proposal", "")
        # Debug: display the proposal fetched from the backend
        st.write("DEBUG: Backend returned proposal:", final_proposal_text)
    else:
        st.error("Failed to fetch the latest proposal.")
        final_proposal_text = ""
    
    if not final_proposal_text.strip():
        st.error("❌ No final proposal found. Please refine or generate the proposal first.")
    else:
        # Update session state with the fetched proposal
        st.session_state.current_proposal = final_proposal_text
        # Create the PDF using the latest refined proposal
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        
        # Use a Unicode font (ensure the file exists in fonts/DejaVuSans.ttf)
        font_path = "fonts/DejaVuSans.ttf"
        pdf.add_font("DejaVu", "", font_path, uni=True)
        pdf.set_font("DejaVu", "", 11)
        pdf.set_left_margin(10)
        pdf.set_right_margin(10)
        formatted_text = "\n".join(
            line.strip() for line in final_proposal_text.split("\n") if line.strip()
        )
        pdf.multi_cell(0, 8, txt=formatted_text, border=0)
        pdf_output = bytes(pdf.output(dest="S"))
        
        st.download_button(
            label="📥 Download Proposal PDF",
            data=pdf_output,
            file_name=f"final_proposal_{int(time.time())}.pdf",  # unique filename to avoid caching issues
            mime="application/pdf"
        )
