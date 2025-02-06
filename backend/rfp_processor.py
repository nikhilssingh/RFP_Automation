# backend/rfp_processor.py

import math
import logging
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from backend.parse_rfp_pdf import parse_rfp_pdf
from backend.llm_utils import llm  
import os
from typing import List


logging.basicConfig(level=logging.INFO)

def compute_rfp_complexity(rfp_text: str, past_rfp_folder: str = "past_rfps ") -> int:
    """
    Compute a complexity score for the RFP using content length, objectives, keywords, and semantic similarity.
    Optionally compares with past RFPs to assess relative complexity.
    Returns a score from 1 (low complexity) to 5 (high complexity).
    """
    try:
        length_score, objectives_score, specificity_score, semantic_score = 0, 0, 0, 0

        # ✅ 1. Length-Based Scoring
        word_count = len(rfp_text.split())
        length_score = math.ceil(min((word_count / 500) * 5 / 5, 5))  
        logging.info(f"Length Score: {length_score}")

        # ✅ 2. Objectives-Based Scoring
        objective_count = rfp_text.lower().count("objective")
        objectives_score = min(objective_count, 5)
        logging.info(f"Objectives Score: {objectives_score}")

        # ✅ 3. Specificity-Based Scoring
        specificity_keywords = ["timeline", "budget", "technical", "deliverables", "integration"]
        specificity_score = sum(1 for word in specificity_keywords if word in rfp_text.lower())
        specificity_score = min(specificity_score, 5)
        logging.info(f"Specificity Score: {specificity_score}")

        # ✅ 4. Extract Past RFPs from PDFs
        past_rfps = []
        if os.path.exists(past_rfp_folder):
            for file in os.listdir(past_rfp_folder):
                if file.endswith(".pdf"):
                    pdf_path = os.path.join(past_rfp_folder, file)
                    extracted_text = parse_rfp_pdf(pdf_path)  # ✅ Ensure this function correctly extracts text
                    if extracted_text:
                        past_rfps.append(extracted_text)
                    else:
                        logging.warning(f"⚠️ Unable to extract text from: {pdf_path}")

        # ✅ 5. Semantic Similarity Scoring (if past RFPs available)
        if past_rfps:
            all_rfps = [rfp_text] + past_rfps
            vectorizer = CountVectorizer().fit_transform(all_rfps)
            vectors = vectorizer.toarray()
            cosine_similarities = cosine_similarity(vectors[0:1], vectors[1:])
            max_similarity = max(cosine_similarities[0]) 
            semantic_score = round(min(max_similarity * 5, 5))
            logging.info(f"Semantic Similarity Score: {semantic_score}")
        else:
            logging.info("⚠️ No past RFPs found in the `past_rfps` folder. Semantic similarity score is 0.")

        # ✅ 6. Weighted Scoring
        weights = {"length": 0.4, "objectives": 0.2, "specificity": 0.2, "semantic": 0.2}
        weighted_score = (
            weights["length"] * length_score +
            weights["objectives"] * objectives_score +
            weights["specificity"] * specificity_score +
            weights["semantic"] * semantic_score
        )
        logging.info(f"Weighted Score (before ceiling): {weighted_score}")

        # ✅ 7. Final Complexity Score
        final_score = max(1, min(math.ceil(weighted_score), 5))
        logging.info(f"Final RFP Complexity Score: {final_score}")

        return final_score

    except Exception as e:
        logging.error(f"❌ Error computing RFP complexity: {e}")
        return 1  # Default to lowest complexity if an error occurs

