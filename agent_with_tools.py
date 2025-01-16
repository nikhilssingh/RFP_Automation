# agent_with_tools.py
import os
from dotenv import load_dotenv
load_dotenv()
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore as LCPinecone
from langchain.agents import initialize_agent, AgentType
from langchain_openai import ChatOpenAI
from langchain.tools import BaseTool
from pydantic import Field
from typing import Any
from embeddings_setup import embeddings

class PineconeSearchTool(BaseTool):
    name: str = "pinecone_retrieval"  
    description: str = "Use this tool to retrieve relevant documents from Pinecone." 
    vector_store: Any = Field(...)  

    def _run(self, query: str) -> str:
        docs = self.vector_store.similarity_search(query, k=2)
        if not docs:  
            return "No relevant documents found."
        
        combined_text = "\n".join([d.page_content for d in docs])
        if "irrelevant" in combined_text.lower():  
            return "Retrieved documents are not relevant to the query."
        return combined_text

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("Async not implemented.")

def categorize_query(query):
    """Categorize the query into predefined use cases."""
    if "RFP" in query or "proposal" in query:
        return "RFP"
    elif "recruitment" in query or "candidate" in query:
        return "Recruitment"
    elif "coding" in query or "evaluation" in query:
        return "Coding Evaluation"
    else:
        return "General"

def get_prompt_for_use_case(use_case, query, retrieved_info):
    """Generate a tailored prompt based on the use case."""
    if use_case == "RFP":
        return f"Using the following retrieved information, draft an RFP response:\n{retrieved_info}\nQuery: {query}"
    elif use_case == "Recruitment":
        return f"Analyze the candidate's data based on this retrieved information:\n{retrieved_info}\nQuery: {query}"
    elif use_case == "Coding Evaluation":
        return f"Evaluate the provided code snippet and provide feedback based on the following retrieved information:\n{retrieved_info}\nQuery: {query}"
    else:  
        return f"Using the following retrieved information, respond to the query:\n{retrieved_info}\nQuery: {query}"

# Function to evaluate and enrich the response
def evaluate_and_enrich_response(retrieved_info: str, llm, question: str) -> str:
    evaluation_prompt = f"""
    Evaluate the following retrieved information for correctness:
    "{retrieved_info}"
    
    - If the information is correct:
        - Confirm its correctness.
        - Expand the response by providing examples, additional options, and explanations.
        - Provide actionable advice for beginners, such as simplified approaches.
    - If the information is incorrect or insufficient:
        - Correct it based on your knowledge.
        - Explain why the retrieved information is wrong or insufficient.
        - Provide a comprehensive answer addressing the query scope, focusing on ease, simplicity, or other relevant factors.
    
    Question: {question}
    """
    enriched_response = llm.invoke(evaluation_prompt)
    return enriched_response.content

def main():
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index_name = "my-hybrid-index"

    index = pc.Index(index_name)

    vectorstore = LCPinecone(index=index, embedding=embeddings)

    pinecone_tool = PineconeSearchTool(vector_store=vectorstore)

    llm = ChatOpenAI(
        openai_api_key=os.environ["OPENAI_API_KEY"],
        model_name="gpt-3.5-turbo",
        temperature=0.0
    )
    
    agent = initialize_agent(
        tools=[pinecone_tool],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )

    # Ask a question to the agent
    questions = ["Create a RFP for a coffee shop that wants to integrate AI in their order mechanism."]
    
    for question in questions:
        retrieved_info = agent.run(question)

        if "No relevant documents found." in retrieved_info:
            fallback_info = "No relevant information found."
            response = evaluate_and_enrich_response(fallback_info, llm, question)
        else:
            response = evaluate_and_enrich_response(retrieved_info, llm, question)

        print("Agent Response:", response)    

if __name__ == "__main__":
    main()
