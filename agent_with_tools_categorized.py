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

# Logging
import logging
logging.basicConfig(level=logging.INFO)


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
    Read and extract all text from the given RFP PDF with error handling.
    Returns the extracted text or an error message.
    """
    try:
        text = ""
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        if not text.strip():
            logging.warning(f"No readable text found in PDF: {pdf_path}")
            return "No readable text found in the PDF."
        return text.strip()
    except FileNotFoundError:
        logging.error(f"File not found: {pdf_path}")
        return "Error: File not found."
    except Exception as e:
        logging.error(f"Error reading PDF {pdf_path}: {e}")
        return f"Error: Unable to process the PDF due to {str(e)}."
    
def summarize_rfp(llm, text: str) -> str:
    """
    Summarize the retrieved response to RFP documents using the LLM with error handling.
    """
    try:
        prompt = f"""
        You are an expert at summarizing response to RFP documents.
        Please summarize the following RFP text in a concise yet comprehensive way:

        {text}
        """
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        logging.error(f"Error during LLM summarization: {e}")
        return "Error: Unable to generate summary."

def expand_rfp(llm, summary: str) -> str:
    """
    Expand the summary into a comprehensive proposal with error handling.
    """
    try:
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
    except Exception as e:
        logging.error(f"Error during LLM expansion: {e}")
        return "Error: Unable to expand summary into a proposal."

def save_rfp_as_pdf(rfp_text: str, pdf_filename="final_rfp.pdf"):
    """
    Save the final RFP text as a PDF with error handling.
    """
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)

        pdf.add_font("DejaVu", "", "DejaVuSansCondensed.ttf", uni=True)
        pdf.set_font("DejaVu", "", 12)

        paragraphs = rfp_text.split("\n\n")
        for para in paragraphs:
            pdf.multi_cell(0, 7, para.strip())
            pdf.ln()

        pdf.output(pdf_filename)
        logging.info(f"PDF saved as '{pdf_filename}'.")
    except Exception as e:
        logging.error(f"Error saving PDF '{pdf_filename}': {e}")

def main():
    try:
        llm = ChatOpenAI(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            model_name="gpt-4o-mini",
            temperature=0.0
        )

        rfp_pdf_path = "user_rfp.pdf"
        rfp_text = parse_rfp_pdf(rfp_pdf_path)
        if "Error:" in rfp_text:
            logging.error(f"Terminating process due to PDF error: {rfp_text}")
            return

        rfp_summary = summarize_rfp(llm, rfp_text)
        if "Error:" in rfp_summary:
            logging.error(f"Terminating process due to summarization error: {rfp_summary}")
            return

        final_rfp_text = expand_rfp(llm, rfp_summary)
        if "Error:" in final_rfp_text:
            logging.error(f"Terminating process due to expansion error: {final_rfp_text}")
            return

        save_rfp_as_pdf(final_rfp_text, pdf_filename="response_to_rfp.pdf")
        logging.info("Response to RFP processing completed successfully.")
    except Exception as e:
        logging.error(f"Unexpected error in main: {e}")

if __name__ == "__main__":
    main()
