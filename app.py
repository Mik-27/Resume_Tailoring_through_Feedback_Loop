from fastapi import FastAPI, HTTPException

from src.main import ResumeAgent

app = FastAPI()

@app.post("/tailor_resume")
async def tailor_resume(resume: str, job_description: str):
    try:
        agent = ResumeAgent()
        response = agent.run(resume, job_description)
        return {"final_resume": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
