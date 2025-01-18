# agent_with_tools_categorized.py
import os
from dotenv import load_dotenv

load_dotenv()

# Pinecone / VectorStore
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore as LCPinecone

# LangChain Agents & Tools
from langchain.agents import initialize_agent, AgentType
from langchain_openai import ChatOpenAI
from langchain.tools import BaseTool
from pydantic import Field
from typing import Any

from embeddings_setup import embeddings

# PDF generation
from fpdf import FPDF
import PyPDF2



class PineconeSearchTool(BaseTool):
    """
    Retrieves relevant documents from Pinecone. 
    We will then summarize these docs with a separate function or chain.
    """
    name: str = "pinecone_retrieval"
    description: str = "Use this tool to retrieve relevant response to RFP documents from Pinecone."
    vector_store: Any = Field(...)

    def _run(self, query: str) -> str:
        docs = self.vector_store.similarity_search(query, k=2)
        if not docs:
            return "No relevant documents found."

        combined_text = "\n".join([d.page_content for d in docs])
        return combined_text

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("Async not implemented.")

def parse_rfp_pdf(pdf_path: str) -> str:
    """
    Read and extract all text from the given RFP PDF.
    Returns the extracted text as a single string.
    """
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            # Extract text from each page and add a newline
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()
def summarize_rfp(llm, text: str) -> str:
    """
    Summarize the retrieved response to RFP documents using the LLM.
    Keep it short, clear, and focused on the main points.
    """
    prompt = f"""
    You are an expert at summarizing response to RFP documents.
    Please summarize the following RFP text in a concise yet comprehensive way:

    {text}
    """
    response = llm.invoke(prompt)
    return response.content.strip()

def expand_rfp(llm, summary: str) -> str:
    prompt = f"""
    Below is a summarized RFP. Please expand or refine this summary into a 
    full and comprehensive proposal/response to the RFP. Use professional 
    language, include critical sections (like objectives, timelines, 
    budgets, and any relevant technical details).

    Summarized RFP Content:
    {summary}
    """
    response = llm.invoke(prompt)
    return response.content.strip()

def save_rfp_as_pdf(rfp_text: str, pdf_filename="final_rfp.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # 1) Register a TTF font with 'uni=True'
    pdf.add_font("DejaVu", "", "DejaVuSansCondensed.ttf", uni=True)
    # 2) Use that font
    pdf.set_font("DejaVu", "", 12)

    paragraphs = rfp_text.split("\n\n")
    for para in paragraphs:
        pdf.multi_cell(0, 7, para.strip())
        pdf.ln()

    pdf.output(pdf_filename)
    print(f"PDF saved as '{pdf_filename}'.")

###############################################################################
# 5) Main
###############################################################################
def main():
    # 1) Load the Large Language Model
    llm = ChatOpenAI(
        openai_api_key=os.environ["OPENAI_API_KEY"],
        # Use your desired model here, e.g., "gpt-4" if you have access
        model_name="gpt-4o-mini",
        temperature=0.0
    )

    # 2) Specify the path to the submitted RFP PDF
    #    In practice, you would get this path from user input, a web form, etc.
    rfp_pdf_path = "user_rfp.pdf"  # Replace with the actual file path

    # 3) Parse the PDF to extract RFP text
    rfp_text = parse_rfp_pdf(rfp_pdf_path)

    # 4) Summarize the RFP
    rfp_summary = summarize_rfp(llm, rfp_text)

    # 5) Expand the summary into a comprehensive proposal/response
    final_rfp_text = expand_rfp(llm, rfp_summary)

    # 6) Save the final response as a PDF
    save_rfp_as_pdf(final_rfp_text, pdf_filename="response_to_rfp.pdf")

    print("\n----- Done. The response to RFP has been generated and saved as PDF. -----\n")

if __name__ == "__main__":
    main()
