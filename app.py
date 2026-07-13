from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import joblib
import pandas as pd
import uvicorn

from database import SessionLocal, engine
from models import Base, PredictionLog

app = FastAPI(title="Alzheimer's Disease Predictor")

# Create tables if they don't exist
Base.metadata.create_all(bind=engine)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load ML model
try:
    model = joblib.load("models/RandomForest_Alzheimers_model.pkl")
except FileNotFoundError:
    print("Model file not found!")
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
            context={"error": "Model not loaded."}
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
        # Predict
        probabilities = model.predict_proba(input_data)[0]

        # Convert NumPy float to Python float
        positive_prob = float(probabilities[1])

        probability_percent = round(positive_prob * 100, 2)

        if positive_prob >= 0.35:
            prediction_result = "High Risk (Positive)"
            result_class = "danger"
        else:
            prediction_result = "Low Risk (Negative)"
            result_class = "success"

        # Save to PostgreSQL
        db = SessionLocal()

        try:
            prediction = PredictionLog(
                FunctionalAssessment=FunctionalAssessment,
                ADL=ADL,
                MemoryComplaints=MemoryComplaints,
                MMSE=MMSE,
                BehavioralProblems=BehavioralProblems,
                SleepQuality=SleepQuality,
                CholesterolHDL=CholesterolHDL,
                prediction=prediction_result,
                probability=probability_percent
            )

            db.add(prediction)
            db.commit()

        except Exception:
            db.rollback()
            raise

        finally:
            db.close()

        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "prediction": prediction_result,
                "probability": probability_percent,
                "result_class": result_class
            }
        )

    except Exception as e:
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "error": f"Prediction Error: {str(e)}"
            }
        )


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )