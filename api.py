from fastapi import FastAPI
from pydantic import BaseModel
from groq import Groq
from rag import load_knowledge_base, get_relevant_info
from multi_agent import run_multi_agent
from evaluator import run_evaluation

app = FastAPI(
    title="Health AI Explainer API",
    description="API for explaining medical test results using AI",
    version="1.0.0"
)

# Load knowledge base once when API starts
knowledge_base = load_knowledge_base()

# Your Groq API key
import os
API_KEY = os.getenv("GROQ_API_KEY")

# Define what the request should look like
class HealthRequest(BaseModel):
    health_data: str
    language: str = "English"

# Define what the response will look like
class HealthResponse(BaseModel):
    extracted_values: str
    safety_assessment: str
    report: str
    quality_review: str
    safety_score: float
    ai_judge_scores: str

# Endpoint 1 - Check if API is running
@app.get("/health")
def health_check():
    return {"status": "running", "message": "Health AI API is online"}

# Endpoint 2 - Explain blood test results
@app.post("/explain", response_model=HealthResponse)
def explain_results(request: HealthRequest):
    client = Groq(api_key=API_KEY)
    
    # Run multi-agent analysis
    result = run_multi_agent(
        request.health_data,
        knowledge_base,
        request.language,
        API_KEY
    )
    
    # Run evaluation
    evaluation = run_evaluation(
        request.health_data,
        result["report"],
        request.language,
        API_KEY
    )
    
    return HealthResponse(
        extracted_values=result["extracted"],
        safety_assessment=result["safety"],
        report=result["report"],
        quality_review=result["review"],
        safety_score=evaluation["safety_score"],
        ai_judge_scores=evaluation["llm_scores"]
    )