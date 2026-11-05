import logging
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .llm import LLMError
from .schemas import ChatRequest, ChatResponse
from .service import answer_question

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("support")

app = FastAPI(title="SaaS Support Assistant")
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


def error_response(code: str, message: str, status: int) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message}})


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    first = (exc.errors() or [{}])[0]
    field = ".".join(str(p) for p in first.get("loc", []) if p != "body")
    msg = str(first.get("msg", "Invalid request")).removeprefix("Value error, ")
    return error_response("invalid_request", f"{field}: {msg}" if field else msg, 422)


@app.exception_handler(LLMError)
async def llm_handler(request: Request, exc: LLMError):
    return error_response(exc.code, exc.public_message, exc.http_status)


@app.exception_handler(Exception)
async def unexpected_handler(request: Request, exc: Exception):
    log.exception("unhandled error")  # full trace stays in server logs only
    return error_response("internal_error", "Something went wrong. Please try again.", 500)


# Sync endpoint: FastAPI runs it in a threadpool so the blocking LLM SDK call doesn't stall the loop.
@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    return answer_question(req)


@app.get("/health")
def health():
    return {"status": "ok"}


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")
