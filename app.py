from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import joblib
import pandas as pd
import uvicorn

app = FastAPI(title="Alzheimer's Disease Predictor")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

try:
    model = joblib.load("models/RandomForest_Alzheimers_model.pkl")
except FileNotFoundError:
    print("Error: Model file not found. Please run main.py first to generate alzheimers_model.pkl")
    model = None

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request, 
        name="index.html", 
        context={"prediction": None}
    )

@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    FunctionalAssessment: float = Form(...),
    ADL: float = Form(...),
    MemoryComplaints: int = Form(...),
    MMSE: float = Form(...),
    BehavioralProblems: int = Form(...),
    SleepQuality: float = Form(...),
    CholesterolHDL: float = Form(...)
):
    if model is None:
        return templates.TemplateResponse(
            request=request, 
            name="index.html", 
            context={"error": "Model not loaded. Please train the model first."}
        )

    input_data = pd.DataFrame([{
        "FunctionalAssessment": FunctionalAssessment,
        "ADL": ADL,
        "MemoryComplaints": MemoryComplaints,
        "MMSE": MMSE,
        "BehavioralProblems": BehavioralProblems,
        "SleepQuality": SleepQuality,
        "CholesterolHDL": CholesterolHDL
    }])

    try:
        probabilities = model.predict_proba(input_data)[0]
        positive_prob = probabilities[1]
        
        if positive_prob >= 0.35:
            prediction_result = "High Risk (Positive)"
            result_class = "danger"
        else:
            prediction_result = "Low Risk (Negative)"
            result_class = "success"

        return templates.TemplateResponse(
            request=request, 
            name="index.html", 
            context={
                "prediction": prediction_result,
                "probability": round(positive_prob * 100, 2),
                "result_class": result_class
            }
        )
        
    except Exception as e:
        return templates.TemplateResponse(
            request=request, 
            name="index.html", 
            context={"error": f"Prediction error: {str(e)}"}
        )

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)