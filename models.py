from sqlalchemy import Column, Integer, Float, String, DateTime
from sqlalchemy.sql import func

from database import Base

class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)

    FunctionalAssessment = Column(Float)
    ADL = Column(Float)
    MemoryComplaints = Column(Integer)
    MMSE = Column(Float)
    BehavioralProblems = Column(Integer)
    SleepQuality = Column(Float)
    CholesterolHDL = Column(Float)

    prediction = Column(String)
    probability = Column(Float)

    created_at = Column(DateTime(timezone=True), server_default=func.now())