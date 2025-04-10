# legacy_code.py

import os
from dotenv import load_dotenv

load_dotenv()

# Pinecone / VectorStore
from pinecone import Pinecone, ServerlessSpec, Index
from langchain_pinecone import PineconeVectorStore as LCPinecone
from langchain_pinecone import PineconeVectorStore

# LangChain Agents & Tools
from langchain.agents import initialize_agent, AgentType
from langchain_openai import ChatOpenAI
from langchain.tools import BaseTool
from pydantic import Field
from typing import Any
import anthropic

from embeddings_setup import embeddings

# PDF generation
from fpdf import FPDF
import PyPDF2

# Logging
import logging
logging.basicConfig(level=logging.INFO)

# Complexity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Performance
import time

# Maths
import math


# Initialize Pinecone Vector Store
index_name = "my-hybrid-index"  
index = Index(
    api_key=os.getenv("PINECONE_API_KEY"),
    host=os.getenv("PINECONE_HOST"),
    name=index_name
)
vectorstore = LCPinecone(index=index, embedding=embeddings)

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
    
def load_past_rfps(folder_path: str) -> list:
    """
    Loads and processes past RFPs from the specified folder.
    Returns a list of RFP texts. Handles errors gracefully.
    """
    rfp_texts = []
    supported_extensions = ('.txt', '.pdf')

    try:
        if not os.path.exists(folder_path):
            logging.error(f"Folder not found: {folder_path}")
            return []

        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)

            if not os.path.isfile(file_path):
                logging.warning(f"Skipping non-file item: {file_name}")
                continue

            if file_name.lower().endswith('.txt'):
                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        rfp_texts.append(file.read())
                        logging.info(f"Successfully loaded text file: {file_name}")
                except Exception as e:
                    logging.error(f"Error reading text file {file_name}: {e}")

            elif file_name.lower().endswith('.pdf'):
                try:
                    rfp_text = parse_rfp_pdf(file_path)  
                    if rfp_text:
                        rfp_texts.append(rfp_text)
                        logging.info(f"Successfully loaded PDF file: {file_name}")
                    else:
                        logging.warning(f"No text found in PDF file: {file_name}")
                except Exception as e:
                    logging.error(f"Error reading PDF file {file_name}: {e}")

            else:
                logging.warning(f"Unsupported file type for: {file_name}. Skipping.")

        return rfp_texts

    except Exception as e:
        logging.error(f"Error accessing or processing folder {folder_path}: {e}")
        return []
    
def compute_rfp_complexity(rfp_text: str, past_rfps: list = None) -> int:
    """
    Compute a complexity score for the RFP using content length, objectives, keywords, and semantic similarity.
    Optionally compares with past RFPs to assess relative complexity.
    Returns a score from 1 (low complexity) to 5 (high complexity).
    """
    try:
        # Initialize all scores to zero
        length_score, objectives_score, specificity_score, semantic_score = 0, 0, 0, 0

        # 1. Length-Based Scoring
        word_count = len(rfp_text.split())
        length_score = math.ceil(min((word_count / 500) * 5 / 5, 5))  # Scales linearly and caps at 5
        logging.info(f"Length Score: {length_score}")

        # 2. Objectives-Based Scoring
        objective_count = rfp_text.lower().count("objective")
        objectives_score = min(objective_count, 5)  # Cap score at 5
        logging.info(f"Objectives Score: {objectives_score}")

        # 3. Specificity-Based Scoring
        specificity_keywords = ["timeline", "budget", "technical", "deliverables", "integration"]
        specificity_score = sum(1 for word in specificity_keywords if word in rfp_text.lower())
        specificity_score = min(specificity_score, 5)  # Cap score at 5
        logging.info(f"Specificity Score: {specificity_score}")

        # 4. Semantic Similarity (if past RFPs provided)
        if past_rfps:
            # Combine the new RFP with past RFPs
            all_rfps = [rfp_text] + past_rfps

            # Create vectorized representations
            vectorizer = CountVectorizer().fit_transform(all_rfps)
            vectors = vectorizer.toarray()

            # Compute cosine similarity of the new RFP with past RFPs
            cosine_similarities = cosine_similarity(vectors[0:1], vectors[1:])
            max_similarity = max(cosine_similarities[0])  # Highest similarity score

            # Rescale semantic score to 1-5
            semantic_score = round(min(max_similarity * 5, 5))
            logging.info(f"Semantic Similarity Score: {semantic_score}")
        else:
            logging.info("No past RFPs provided. Semantic similarity score is 0.")

        # Weighted Scoring (adjust weights for each factor)
        weights = {"length": 0.4, "objectives": 0.2, "specificity": 0.2, "semantic": 0.2}
        weighted_score = (
            weights["length"] * length_score +
            weights["objectives"] * objectives_score +
            weights["specificity"] * specificity_score +
            weights["semantic"] * semantic_score
        )
        logging.info(f"Weighted Score (before ceiling): {weighted_score}")

        # Final Score (round to nearest integer within range 1-5)
        final_score = max(1, min(math.ceil(weighted_score), 5))
        logging.info(f"Final RFP Complexity Score: {final_score}")

        return final_score

    except Exception as e:
        logging.error(f"Error computing RFP complexity: {e}")
        return 1  # Default to low complexity

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
    
def summarize_rfp(llm, rfp_text: str, retrieved_docs: str, complexity: int) -> str:
    """
    Summarize the new RFP with an emphasis on its content and use retrieved documents as context and use the retrieved documents to identify 
    common themes or successful practices and integrate them where relevant.
    Also, use the retrieved documents for style and structure reference.
    Ensure the summary is concise, professional, and free from unnecessary repetition.
    Adjust the level of detail based on the complexity score.
    """
    try:
        if complexity >= 4:
            detail_level = "detailed"
        elif complexity == 3:
            detail_level = "balanced"
        else:
            detail_level = "concise"

        prompt = f"""
        You are an expert at summarizing RFP documents. Provide a {detail_level} summary 
        focusing on the following new RFP with an emphasis on its content. Use the retrieved documents as additional context to identify 
        common themes or successful practices and integrate them where relevant:

        New RFP:
        {rfp_text}

        Context (Old Proposals):
        {retrieved_docs}
        """
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        logging.error(f"Error during LLM summarization: {e}")
        return "Error: Unable to generate summary."

def expand_rfp(llm, summary: str, complexity: int) -> str:
    """
    Expand the summary into a comprehensive proposal with adjustments for complexity.
    """
    try:
        if complexity >= 4:
            detail_level = "thorough"
        elif complexity == 3:
            detail_level = "balanced"
        else:
            detail_level = "basic"

        prompt = f"""
        Below is a summarized RFP. Expand or refine this summary into a {detail_level} proposal 
        with professional language, objectives, timelines, budgets, and technical details.

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
        
def save_draft_for_review(content: str, filename: str) -> None:
    """
    Save the generated content as a draft for human review.
    """
    try:
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(content)
        logging.info(f"Draft saved for review: {filename}")
    except Exception as e:
        logging.error(f"Error saving draft for review {filename}: {e}")


def load_reviewed_content(filename: str) -> str:
    """
    Load the reviewed content after human review.
    """
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            reviewed_content = file.read()
        logging.info(f"Reviewed content loaded from {filename}")
        return reviewed_content
    except Exception as e:
        logging.error(f"Error loading reviewed content from {filename}: {e}")
        return "Error: Unable to load reviewed content."

def main():
    try:
        llm = ChatOpenAI(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            model_name="gpt-4o-mini",
            temperature=0.0
        )
        
        # Track start time for the process
        process_start_time = time.time()
        
        rfp_pdf_path = "user_rfp.pdf"
        rfp_text = parse_rfp_pdf(rfp_pdf_path)
        if "Error:" in rfp_text:
            logging.error(f"Terminating process due to PDF error: {rfp_text}")
            return
        
        # Load past RFPs
        past_rfps_folder = "past_rfps" 
        past_rfps = load_past_rfps(past_rfps_folder)
        if not past_rfps:
            logging.warning("No past RFPs loaded. Complexity scoring will omit semantic similarity.")

        complexity_score = compute_rfp_complexity(rfp_text,past_rfps)
        logging.info(f"Complexity Score Computed: {complexity_score}")
        
        retrieval_tool = PineconeSearchTool(vector_store=vectorstore)

        # Retrieve related proposals
        retrieval_start = time.time()
        retrieved_docs = retrieval_tool._run(rfp_text)
        retrieval_time = time.time() - retrieval_start
        logging.info(f"Document Retrieval Time: {retrieval_time:.2f} seconds")

        # Combine retrieved content with the RFP text
        combined_rfp_text = f"{retrieved_docs}\n\n{rfp_text}"
        
        # Summarization Step
        summarization_start = time.time()
        rfp_summary = summarize_rfp(
            llm=llm,
            rfp_text=rfp_text,
            retrieved_docs=retrieved_docs,
            complexity=complexity_score
        )
        summarization_time = time.time() - summarization_start
        logging.info(f"Summarization Time: {summarization_time:.2f} seconds")
        if "Error:" in rfp_summary:
            logging.error(f"Terminating process due to summarization error: {rfp_summary}")
            return
        logging.info("Summarization successful.")
        
        # Save Summary for Review
        summary_draft_file = "summary_draft.txt"
        save_draft_for_review(rfp_summary, summary_draft_file)
        logging.info(f"Waiting for human review of the summary saved in {summary_draft_file}.")
        input("Press Enter once the summary review is complete and saved.")  # Pause for review
        rfp_summary = load_reviewed_content(summary_draft_file)
        if "Error:" in rfp_summary:
            logging.error(f"Terminating process due to error in loading reviewed summary: {rfp_summary}")
            return

        # Proposal Generation Step
        proposal_start = time.time()
        final_rfp_text = expand_rfp(llm, f"{retrieved_docs}\n\n{rfp_summary}", complexity_score)
        proposal_time = time.time() - proposal_start
        logging.info(f"Proposal Generation Time: {proposal_time:.2f} seconds")
        if "Error:" in final_rfp_text:
            logging.error(f"Terminating process due to expansion error: {final_rfp_text}")
            return
        logging.info("Proposal generation successful.")

        # Save Proposal for Review
        proposal_draft_file = "proposal_draft.txt"
        save_draft_for_review(final_rfp_text, proposal_draft_file)
        logging.info(f"Waiting for human review of the proposal saved in {proposal_draft_file}.")
        input("Press Enter once the proposal review is complete and saved.")  # Pause for review
        final_rfp_text = load_reviewed_content(proposal_draft_file)
        if "Error:" in final_rfp_text:
            logging.error(f"Terminating process due to error in loading reviewed proposal: {final_rfp_text}")
            return
        
        save_rfp_as_pdf(final_rfp_text, pdf_filename="response_to_rfp.pdf")
        logging.info("Final proposal saved as PDF successfully.")
        
        # Log total process time
        total_process_time = time.time() - process_start_time
        logging.info(f"Total Process Time: {total_process_time:.2f} seconds")
        
    except Exception as e:
        logging.error(f"Unexpected error in main: {e}")

if __name__ == "__main__":
    main()