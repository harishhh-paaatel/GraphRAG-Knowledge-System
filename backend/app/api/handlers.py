"""Exception handlers for answer generation.

The endpoint (a later milestone) maps generation failures itself so it can attach the retrieved
sources. This handler is the safety net: any ``GenerationError`` that escapes still becomes the
standard error envelope with a stable code, never a stack trace or a provider response body.
"""

from fastapi import Request
from fastapi.responses import JSONResponse

from app.services.generation.errors import GenerationError, to_app_error


async def generation_error_handler(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, GenerationError)
    err = to_app_error(exc)
    return JSONResponse(
        {"error": {"code": err.code, "message": err.message}}, status_code=err.status_code
    )
