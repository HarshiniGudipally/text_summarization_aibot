from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from backend.text_summarization import summarize_text
from backend.database import store_summary, get_summary_history
import logger

app = FastAPI(debug=True)

class SummarizationRequest(BaseModel):
    text: str
    length: str
    method: str

@app.post("/summarize")
async def summarize(request: SummarizationRequest):
    try:
        summary = summarize_text(request.text, request.length, request.method)
        await store_summary(request.text, summary, request.length, request.method)
        return {"summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history")
async def get_history():
    try:
        history = await get_summary_history()
        return {"history": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred."}
    )
