from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import os
import uvicorn
from engine import (
    InvestmentConstraints,
    GeoInvestPortfolioResponse,
    calculate_dynamic_portfolio_analysis
)

app = FastAPI(
    title="Geo-Invest AI Engine API",
    description="Dynamic LLM-based Location Intelligence & Feasibility Analyzer Backend",
    version="3.0.0"
)

# Enable CORS for cross-origin frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/v2/optimize-portfolio", response_model=GeoInvestPortfolioResponse)
async def optimize_portfolio(constraints: InvestmentConstraints):
    """
    Executes dynamic location feasibility analysis and ROI estimation using Gemini.
    """
    try:
        print(f"[API] Analyzing feasibility for prompt: '{constraints.user_prompt}'")
        print(f"      No Business Idea: {constraints.no_business_idea}")
        
        # Trigger dynamic Gemini optimization and mapping
        response_payload = calculate_dynamic_portfolio_analysis(constraints)
        
        print(f"[API] Analysis completed successfully. Evolved {len(response_payload.recommended_locations)} site recommendations.")
        return response_payload
        
    except ValueError as ve:
        print(f"[API Error] Validation failure: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        import traceback
        print(f"[API Error] LLM generation failed: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Dynamic portfolio analysis failed: {str(e)}")

@app.get("/", response_class=HTMLResponse)
async def get_index():
    """
    Serves the premium Location Intelligence Dashboard at root address.
    """
    index_path = os.path.join(os.path.dirname(__file__), "index.html")
    if not os.path.exists(index_path):
        return HTMLResponse("<h1>index.html not found! Please compile the dashboard frontend.</h1>", status_code=404)
    with open(index_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)

if __name__ == "__main__":
    print("[Server] Starting Geo-Invest AI API Server on http://localhost:8000")
    uvicorn.run("server:app", host="0.0.0.0", port=8000)
