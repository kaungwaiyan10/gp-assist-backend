
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai
import os

app = FastAPI()

# Allow frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

openai.api_key = os.getenv("OPENAI_API_KEY")

class PatientData(BaseModel):
    patientId: str
    history: str
    physicalExam: str
    investigations: str
    imaging: str

@app.post("/analyze")
async def analyze(data: PatientData):
    prompt = (
        f"Patient ID: {data.patientId}\n"
        f"History: {data.history}\n"
        f"Physical Exam: {data.physicalExam}\n"
        f"Investigations: {data.investigations}\n"
        f"Imaging: {data.imaging}\n\n"
        "Based on the above, suggest likely differential diagnoses and a management plan."
    )
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful medical assistant for general practitioners."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=600,
    )
    return {"diagnosis": response.choices[0].message.content}
