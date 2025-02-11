# backend/llm_utils.py
import os
from dotenv import load_dotenv
load_dotenv()
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain

openai_api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(
    openai_api_key=openai_api_key,
    model_name="gpt-4o-mini",
    temperature=0.0
)

def expand_rfp(rfp_text, retrieved_docs):
    """Generates a thorough business proposal in response to an RFP, ensuring past proposal data is effectively reused."""
    
    structured_context = "\n\n".join([
        f"🔹 **Reference Proposal {i+1}**:\n{doc}" for i, doc in enumerate(retrieved_docs)
    ]) if retrieved_docs else "No similar documents found."
        
    prompt = f"""
    You are a professional business consultant responding to a client’s RFP. Your task is to generate a **thorough business proposal** that directly addresses the client's needs.

    ---
    **📜 Client’s RFP to Respond To:**
    {rfp_text}

    ---
    **📂 Past Successful Proposals (USE THESE TO SHAPE THE RESPONSE and fill in the company name, contact information, etc. and structure which is redundant from the past proposals):**
    {structured_context}

    ---
    **Proposal Format:**
    
    📌 **Cover Letter**  
    - Start with a compelling opening that differentiates us.  
    - Showcase our expertise and success in similar projects.  
    - End with a warm call to action.  

    📌 **Understanding of Client Needs**  
    - Identify key challenges mentioned in the RFP.  
    - Use retrieved proposals to match solutions to the client’s goals.  

    📌 **Proposed Solution**  
    - Tailor the response using **retrieved past proposals** (inventory optimization, customer recommendations, etc.).  
    - Clearly describe the AI-driven enhancements.  

    📌 **Project Plan & Implementation Timeline**  
    - **Assign team members** to each phase for credibility.  
    - Provide detailed milestones and clear deliverables.  

    📌 **Pricing & Payment Terms**  
    - Extract competitive pricing from past proposals.  
    - Justify the investment with **ROI-driven language**.  

    📌 **Technical Approach**  
    - Explain AI models, data processing, and security measures.  

    📌 **Company Experience**  
    - Highlight **measurable successes** from past projects.  
    - Include relevant testimonials and case studies.  

    📌 **Case Studies & Testimonials**  
    - Use real success stories with **quantifiable impact** (e.g., 20% increase in efficiency).  

    📌 **Conclusion & Call to Action**  
    - End with a clear **next step** (e.g., scheduling a consultation call).  
    - Ensure persuasive, client-centered writing.  

    🎯 **Important:**  
    - Reference **retrieved proposals** in relevant sections.  
    - Ensure pricing, solutions, and technical details align with industry best practices.  
    """

    print(f"\n📝 Sending this prompt to GPT:\n{prompt[:1500]}")  # ✅ Debugging output

    response = llm.invoke(prompt)
    return response.content.strip()


# Maintain memory for ongoing refinements
conversation_memory = {"latest_proposal": ""}

def refine_proposal(existing_proposal: str, user_feedback: str) -> str:
    """
    Refines the latest generated proposal based on user feedback.
    """
    current_proposal = conversation_memory.get("latest_proposal", "")
    
    if not current_proposal:
        return "No proposal available. Please generate one first."
    
    prompt = f"""
    You are an expert proposal writer refining a business proposal based on user feedback.
    
    ---
    **Current Proposal:**
    {current_proposal}
    
    **User Feedback:**
    {user_feedback}
    
    **Instructions:**
    - Improve clarity, conciseness, and persuasiveness.
    - Address specific concerns raised by the user.
    - Maintain a professional and structured format.
    """
    
    response = llm.invoke(prompt)
    
    # 🔄 Update memory with refined proposal
    refined_proposal = response.content.strip()
    conversation_memory["latest_proposal"] = refined_proposal
    
    return refined_proposal
