# routes/proposal_routes.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from backend.llm_utils import expand_rfp
from backend.pinecone_utils import retrieve_similar_docs
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()
import os

proposal_router = APIRouter()

# ✅ Initialize GPT-4o
llm = ChatOpenAI(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model_name="gpt-4o-mini",
    temperature=0.0
)

# ✅ Define request model
class RFPRequest(BaseModel):
    rfp_text: str

@proposal_router.post("/generate_proposal")
def generate_proposal(request: RFPRequest):
    """ Generate a proposal in response to an RFP while leveraging retrieved documents for RAG. """
    try:
        rfp_text = request.rfp_text.strip()
        if not rfp_text:
            raise HTTPException(status_code=400, detail="RFP text cannot be empty.")

        # ✅ Step 1: Retrieve Similar Proposals from Pinecone
        retrieved_docs = retrieve_similar_docs(rfp_text, top_k=3)

        print("\n🔍 **Retrieved Documents for RAG:**\n", retrieved_docs)  # ✅ Debugging

        # ✅ Step 2: Generate a Proposal Based on Full RFP & Retrieved Context
        proposal = expand_rfp(rfp_text, retrieved_docs)

        return {
            "proposal": proposal,
            "retrieved_docs": retrieved_docs  # ✅ Debugging output
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating proposal: {str(e)}")

