from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.main import ResumeAgent

app = FastAPI()

class ApplicationInfo(BaseModel):
    resume: str
    job_description: str

@app.post("/tailor_resume")
async def tailor_resume(app_info: ApplicationInfo):
    try:
        resume = app_info.resume
        job_description = app_info.job_description
        if not resume:
            raise HTTPException(status_code=400, detail="Resume is required")
        if not job_description:
            raise HTTPException(status_code=400, detail="Job Description is required")
        agent = ResumeAgent()
        response = agent.run(resume, job_description)
        return {"final_resume": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
