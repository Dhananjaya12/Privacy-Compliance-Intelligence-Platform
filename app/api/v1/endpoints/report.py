"""
app/api/v1/endpoints/report.py
-------------------------------
POST /api/v1/report/pdf — render a compliance report (Markdown) to a PDF.

The audit endpoint returns the report as Markdown in `answer`; the frontend
posts that Markdown here to get a downloadable PDF. Rendering is Markdown →
HTML → PDF via xhtml2pdf (pure-Python, Windows-friendly).
"""

from __future__ import annotations

import io
import logging

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter(tags=["report"])


class ReportRequest(BaseModel):
    markdown: str = Field(..., description="Compliance report in Markdown")
    filename: str = Field(default="compliance_report.pdf")


_PDF_CSS = """
@page { size: A4; margin: 1.6cm; }
body { font-family: Helvetica, Arial, sans-serif; font-size: 10pt; color: #1a1a1a; }
h1 { font-size: 18pt; color: #1a2b4a; }
h2 { font-size: 13pt; color: #1a2b4a; border-bottom: 1px solid #ccc; padding-bottom: 3px; }
h3 { font-size: 11pt; color: #2a3b5a; }
table { width: 100%; border-collapse: collapse; margin: 8px 0; }
th, td { border: 1px solid #bbb; padding: 4px 6px; font-size: 9pt; text-align: left; }
th { background: #eef2f8; }
li { margin-bottom: 3px; }
code { font-family: Courier, monospace; }
"""


def _markdown_to_html(md_text: str) -> str:
    import markdown as md_lib

    body = md_lib.markdown(md_text, extensions=["tables", "fenced_code", "sane_lists"])
    return f"<html><head><style>{_PDF_CSS}</style></head><body>{body}</body></html>"


@router.post(
    "/report/pdf",
    summary="Render a compliance report to PDF",
    response_class=StreamingResponse,
)
async def report_pdf(body: ReportRequest):
    if not body.markdown.strip():
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Empty report.")

    try:
        from xhtml2pdf import pisa
    except ImportError as exc:  # pragma: no cover
        raise HTTPException(
            status.HTTP_501_NOT_IMPLEMENTED,
            "PDF generation unavailable — install xhtml2pdf and markdown.",
        ) from exc

    html = _markdown_to_html(body.markdown)
    buf = io.BytesIO()
    result = pisa.CreatePDF(src=html, dest=buf)
    if result.err:
        logger.error("xhtml2pdf reported %d errors", result.err)
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "PDF render failed.")

    buf.seek(0)
    filename = body.filename if body.filename.endswith(".pdf") else f"{body.filename}.pdf"
    return StreamingResponse(
        buf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
