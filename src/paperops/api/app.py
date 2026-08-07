"""FastAPI application for persistent PaperOps workflow jobs."""

from __future__ import annotations

import asyncio
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, BinaryIO
from uuid import uuid4

import aiosqlite
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, status
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from paperops.api.models import (
    ApprovalAccepted,
    HealthView,
    JobAccepted,
    JobView,
    ResumeAccepted,
)
from paperops.api.runner import JobAlreadyRunningError, JobRunner
from paperops.clients.fakes import FakeKnowledgeBaseClient, FakeParserClient
from paperops.clients.mineru import MinerUClient
from paperops.clients.protocols import ParserClient, RetrievalBackend
from paperops.clients.ragflow import RAGFlowClient
from paperops.graph import build_graph
from paperops.models import (
    ApprovalAction,
    ApprovalDecision,
    FailureCode,
    JobStatus,
    QualityDecision,
    QualityMetrics,
    QualityVerdict,
    RetrievalReport,
    WorkflowEvent,
    WorkflowFailure,
)
from paperops.retrieval.native import NativeRetrievalBackend
from paperops.settings import Settings

_CHECKPOINT_TYPES = (
    ApprovalAction,
    ApprovalDecision,
    FailureCode,
    JobStatus,
    QualityDecision,
    QualityMetrics,
    QualityVerdict,
    RetrievalReport,
    WorkflowEvent,
    WorkflowFailure,
)


def _checkpoint_serializer() -> JsonPlusSerializer:
    """Allow only the PaperOps types intentionally persisted in checkpoints."""
    return JsonPlusSerializer(allowed_msgpack_modules=_CHECKPOINT_TYPES)


def _copy_pdf_upload(
    source: BinaryIO,
    destination: Path,
    max_bytes: int,
) -> None:
    """Copy one PDF to a controlled path with signature and size checks."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_suffix(".pdf.tmp")
    total_bytes = 0
    try:
        source.seek(0)
        with temporary_path.open("wb") as output:
            first_chunk = source.read(min(1024 * 1024, max_bytes + 1))
            if not first_chunk.startswith(b"%PDF-"):
                raise ValueError("The uploaded file does not have a PDF signature")
            total_bytes += len(first_chunk)
            if total_bytes > max_bytes:
                raise ValueError("The uploaded PDF exceeds the configured size limit")
            output.write(first_chunk)
            while chunk := source.read(1024 * 1024):
                total_bytes += len(chunk)
                if total_bytes > max_bytes:
                    raise ValueError(
                        "The uploaded PDF exceeds the configured size limit"
                    )
                output.write(chunk)
        temporary_path.replace(destination)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def _safe_source_name(filename: str | None) -> str:
    """Keep a readable filename without trusting a client-supplied path."""
    stem = Path(filename or "paper.pdf").stem
    normalized = re.sub(r"[^\w.-]+", "-", stem).strip("-._")[:80]
    windows_reserved_names = {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{number}" for number in range(1, 10)),
        *(f"LPT{number}" for number in range(1, 10)),
    }
    if normalized.upper() in windows_reserved_names:
        normalized = f"paper-{normalized}"
    return f"{normalized or 'paper'}.pdf"


def _build_clients(
    settings: Settings,
) -> tuple[ParserClient, RetrievalBackend]:
    """Create concrete adapters for the selected local or real profile."""
    if settings.client_mode == "real":
        retrieval_backend: RetrievalBackend
        if settings.retrieval_backend == "ragflow":
            retrieval_backend = RAGFlowClient(settings)
        else:
            retrieval_backend = NativeRetrievalBackend(settings)
        return MinerUClient(settings), retrieval_backend
    return (
        FakeParserClient(settings.artifacts_dir),
        FakeKnowledgeBaseClient(),
    )


async def _close_client(client: object) -> None:
    """Close a concrete adapter when it owns external connection pools."""
    close = getattr(client, "aclose", None)
    if close is not None:
        await close()


def create_app(
    *,
    settings: Settings | None = None,
    parser: ParserClient | None = None,
    knowledge_base: RetrievalBackend | None = None,
) -> FastAPI:
    """Create an application with optional fake clients for isolated tests."""
    resolved_settings = settings or Settings()
    if (parser is None) != (knowledge_base is None):
        raise ValueError("parser and knowledge_base must be injected together")

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        await asyncio.to_thread(
            resolved_settings.checkpoint_db.parent.mkdir,
            parents=True,
            exist_ok=True,
        )
        async with aiosqlite.connect(
            resolved_settings.checkpoint_db
        ) as checkpoint_connection:
            checkpointer = AsyncSqliteSaver(
                checkpoint_connection,
                serde=_checkpoint_serializer(),
            )
            await checkpointer.setup()
            active_parser, active_knowledge_base = (
                (parser, knowledge_base)
                if parser is not None and knowledge_base is not None
                else _build_clients(resolved_settings)
            )
            graph = build_graph(
                parser=active_parser,
                knowledge_base=active_knowledge_base,
                settings=resolved_settings,
                checkpointer=checkpointer,
            )
            runner = JobRunner(graph)
            application.state.settings = resolved_settings
            application.state.retrieval_backend_name = active_knowledge_base.name
            application.state.graph = graph
            application.state.runner = runner
            try:
                yield
            finally:
                await runner.shutdown()
                await _close_client(active_parser)
                if active_knowledge_base is not active_parser:
                    await _close_client(active_knowledge_base)

    application = FastAPI(
        title="PaperOps API",
        version="0.1.0",
        lifespan=lifespan,
    )

    @application.get("/health", response_model=HealthView)
    async def health(request: Request) -> HealthView:
        """Report process liveness and local scheduler pressure."""
        runner: JobRunner = request.app.state.runner
        active_settings: Settings = request.app.state.settings
        return HealthView(
            client_mode=active_settings.client_mode,
            retrieval_backend=request.app.state.retrieval_backend_name,
            active_jobs=runner.active_count,
        )

    @application.post(
        "/jobs",
        response_model=JobAccepted,
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def submit_job(
        request: Request,
        target_knowledge_base: str = Form(min_length=1),
        file: UploadFile = File(),
    ) -> JobAccepted:
        """Store one PDF under a controlled path and schedule its workflow."""
        if not file.filename or Path(file.filename).suffix.lower() != ".pdf":
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail="PaperOps accepts PDF uploads only",
            )
        dataset_id = target_knowledge_base.strip()
        if not re.fullmatch(r"[0-9A-Za-z_-]{1,128}", dataset_id):
            raise HTTPException(
                status_code=422,
                detail=(
                    "target_knowledge_base must be a collection id containing "
                    "letters, digits, underscores, or hyphens"
                ),
            )

        thread_id = str(uuid4())
        active_settings: Settings = request.app.state.settings
        destination = (
            active_settings.knowledge_dir
            / "uploads"
            / thread_id
            / _safe_source_name(file.filename)
        )
        try:
            await asyncio.to_thread(
                _copy_pdf_upload,
                file.file,
                destination,
                active_settings.max_upload_bytes,
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=422,
                detail=str(exc),
            ) from exc
        except OSError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to persist the upload: {exc}",
            ) from exc
        finally:
            await file.close()

        runner: JobRunner = request.app.state.runner
        initial_state = {
            "source_pdf": str(destination),
            "target_knowledge_base": dataset_id,
        }
        await request.app.state.graph.aupdate_state(
            JobRunner.config(thread_id),
            initial_state,
            as_node="__start__",
        )
        await runner.schedule(thread_id, None)
        return JobAccepted(
            thread_id=thread_id,
            status=JobStatus.PENDING,
            status_url=f"/jobs/{thread_id}",
        )

    @application.get("/jobs/{thread_id}", response_model=JobView)
    async def get_job(thread_id: str, request: Request) -> JobView:
        """Read the latest durable checkpoint for one workflow thread."""
        return await _job_view(request, thread_id)

    @application.post(
        "/jobs/{thread_id}/approval",
        response_model=ApprovalAccepted,
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def submit_approval(
        thread_id: str,
        decision: ApprovalDecision,
        request: Request,
    ) -> ApprovalAccepted:
        """Resume a human interrupt with one validated approve/reject command."""
        view = await _job_view(request, thread_id)
        if view.running:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="The job is still running",
            )
        if not view.approval_required:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="The job is not waiting for approval",
            )
        runner: JobRunner = request.app.state.runner
        try:
            await runner.schedule(
                thread_id,
                Command(resume=decision.model_dump(mode="json")),
            )
        except JobAlreadyRunningError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(exc),
            ) from exc
        return ApprovalAccepted(
            thread_id=thread_id,
            decision=decision,
            status_url=f"/jobs/{thread_id}",
        )

    @application.post(
        "/jobs/{thread_id}/resume",
        response_model=ResumeAccepted,
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def resume_job(thread_id: str, request: Request) -> ResumeAccepted:
        """Continue a nonterminal checkpoint after a service restart."""
        view = await _job_view(request, thread_id)
        if view.running:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="The job is already running",
            )
        if view.approval_required:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Use the approval endpoint for a human-review interrupt",
            )
        if not view.next_nodes:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="The job has no unfinished checkpoint to resume",
            )
        runner: JobRunner = request.app.state.runner
        try:
            await runner.schedule(thread_id, None)
        except JobAlreadyRunningError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(exc),
            ) from exc
        return ResumeAccepted(
            thread_id=thread_id,
            status_url=f"/jobs/{thread_id}",
        )

    return application


async def _job_view(request: Request, thread_id: str) -> JobView:
    """Map one LangGraph state snapshot to the stable HTTP response model."""
    graph = request.app.state.graph
    runner: JobRunner = request.app.state.runner
    snapshot = await graph.aget_state(JobRunner.config(thread_id))
    values: dict[str, Any] = snapshot.values
    if not values and not runner.is_known(thread_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="PaperOps job not found",
        )
    job_status = values.get("status", JobStatus.PENDING)
    if not isinstance(job_status, JobStatus):
        job_status = JobStatus(job_status)
    has_interrupt = any(task.interrupts for task in snapshot.tasks)
    return JobView(
        thread_id=thread_id,
        job_id=values.get("job_id"),
        status=job_status,
        running=runner.is_running(thread_id),
        approval_required=(job_status == JobStatus.WAITING_APPROVAL and has_interrupt),
        next_nodes=list(snapshot.next),
        parse_attempts=values.get("parse_attempts", 0),
        parsed_markdown_path=values.get("parsed_markdown_path") or None,
        quality_decision=values.get("quality_decision"),
        indexed_document_id=values.get("indexed_document_id") or None,
        indexed_chunk_count=values.get("indexed_chunk_count", 0),
        retrieval_report=values.get("retrieval_report"),
        failure=values.get("failure"),
        events=values.get("events", []),
        runtime_error=runner.runtime_error(thread_id),
    )


app = create_app()


def run() -> None:
    """Run the development API server through the installed console script."""
    uvicorn.run("paperops.api.app:app", host="127.0.0.1", port=8080)
