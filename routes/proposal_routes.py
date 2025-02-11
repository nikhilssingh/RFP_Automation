# routes/proposal_routes.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from backend.llm_utils import expand_rfp, refine_proposal, conversation_memory
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

class RefineRequest(BaseModel):
    user_feedback: str
    
@proposal_router.post("/refine_proposal")
def refine_proposal_endpoint(refine_data: RefineRequest):
    """API endpoint to refine the latest proposal based on user feedback."""
    try:
        user_feedback = refine_data.user_feedback
        if not user_feedback:
            raise HTTPException(status_code=400, detail="User feedback is required.")

        # 🔍 Fetch the latest stored proposal
        latest_proposal = conversation_memory.get("latest_proposal", None)
        if not latest_proposal:
            raise HTTPException(status_code=400, detail="No existing proposal to refine.")

        # ✅ Refine based on stored proposal + user feedback
        refined_proposal_result = refine_proposal(latest_proposal, user_feedback)

        # 🔄 Store the refined version
        conversation_memory["latest_proposal"] = refined_proposal_result

        return {"refined_proposal": refined_proposal_result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error refining proposal: {str(e)}")

class StoreProposalRequest(BaseModel):
    proposal: str

@proposal_router.post("/store_proposal")
def store_proposal_endpoint(proposal: StoreProposalRequest):
    """API endpoint to store the latest generated proposal."""
    conversation_memory["latest_proposal"] = proposal
    return {"message": "Proposal stored successfully."}
