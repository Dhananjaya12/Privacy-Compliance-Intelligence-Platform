import shutil
import tempfile
from pathlib import Path
from typing import List

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from app.schemas import IngestResponse

router = APIRouter(tags=["ingest"])


@router.post("/ingest", response_model=IngestResponse)
async def ingest(files: List[UploadFile] = File(...), request: Request = None):
    if not getattr(request.app.state, "ready", False):
        raise HTTPException(status_code=503, detail="Pipeline still loading.")

    pipeline = request.app.state.pipeline
    config = request.app.state.config

    saved: List[Path] = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for upload in files:
            if not upload.filename.endswith(".pdf"):
                raise HTTPException(status_code=400, detail=f"{upload.filename} is not a PDF.")

            content = await upload.read()
            dest = Path(tmpdir) / upload.filename
            dest.write_bytes(content)

            # Copy into the pipeline's pdf directory
            pdf_dir: Path = pipeline.pdf_dir
            pdf_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(dest, pdf_dir / upload.filename)
            saved.append(pdf_dir / upload.filename)

        # Re-run pipeline steps
        documents = pipeline.extract_text()
        chunks = pipeline.chunk_documents(documents)
        retriever = pipeline.build_vectorstore(chunks)

        # Rebuild agent with fresh retriever
        from agent.graph import build_agent
        request.app.state.agent = build_agent(retriever, pipeline._generator, config)

    return IngestResponse(
        message=f"Ingested {len(saved)} file(s) successfully.",
        files_processed=len(saved),
        chunks_created=len(chunks),
    )
