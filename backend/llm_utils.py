# backend/llm_utils.py
import os
from dotenv import load_dotenv
load_dotenv()
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

openai_api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(
    openai_api_key=openai_api_key,
    model_name="gpt-4o-mini",
    temperature=0.0
)

def expand_rfp(rfp_text, retrieved_docs, complexity):
    """ Generates a business proposal in response to an RFP, ensuring past proposal data is effectively reused. """
    
    structured_context = "\n\n".join([
        f"🔹 **Reference Proposal {i+1}**:\n{doc}" for i, doc in enumerate(retrieved_docs)
    ]) if retrieved_docs else "No similar documents found."

    if complexity >= 4:
        detail_level = "highly detailed and technical"
    elif complexity == 3:
        detail_level = "well-balanced with key details"
    else:
        detail_level = "concise and to the point"
        
    prompt = f"""
    You are a professional business consultant responding to a client’s RFP. Your task is to generate a **{detail_level}** that directly addresses the client's needs.

    ---
    **📜 Client’s RFP to Respond To:**
    {rfp_text}

    ---
    **📂 Past Successful Proposals (USE THESE TO SHAPE THE RESPONSE):**
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
