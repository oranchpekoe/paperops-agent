"""HTTP adapter for the official self-hosted MinerU task API."""

from __future__ import annotations

import asyncio
import json
import mimetypes
import os
import shutil
import stat
import tempfile
import time
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx
from pydantic import BaseModel

from paperops.clients.errors import MinerUError, MinerUTimeout
from paperops.clients.http import require_json_object, response_detail
from paperops.models import ParseRequest, ParseResult
from paperops.settings import Settings


class _TaskManifest(BaseModel):
    """Persist enough MinerU task identity to continue polling after a replay."""

    idempotency_key: str
    task_id: str
    status_url: str
    result_url: str


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Persist a small manifest atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _read_manifest(path: Path) -> _TaskManifest | None:
    """Load a valid task manifest or fail closed on corrupted state."""
    if not path.is_file():
        return None
    try:
        return _TaskManifest.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise MinerUError(f"Invalid MinerU task manifest: {path}") from exc


def _find_markdown(result_dir: Path, source_stem: str) -> Path | None:
    """Select the single source Markdown output from an extracted MinerU result."""
    if not result_dir.is_dir():
        return None
    candidates = [path for path in result_dir.rglob("*.md") if path.is_file()]
    if not candidates:
        return None
    exact = [path for path in candidates if path.stem == source_stem]
    if len(exact) == 1:
        return exact[0]
    if len(candidates) == 1:
        return candidates[0]
    raise MinerUError(
        "MinerU returned multiple Markdown files for one source document: "
        + ", ".join(str(path.relative_to(result_dir)) for path in candidates)
    )


def _safe_extract_result(
    zip_path: Path,
    result_dir: Path,
    *,
    max_extracted_bytes: int,
    source_stem: str,
) -> Path:
    """Extract a MinerU ZIP with traversal, symlink, and size protections."""
    result_dir.parent.mkdir(parents=True, exist_ok=True)
    existing = _find_markdown(result_dir, source_stem)
    if existing is not None:
        return existing

    temporary_dir = Path(
        tempfile.mkdtemp(prefix="mineru-result-", dir=str(result_dir.parent))
    )
    extracted_bytes = 0
    try:
        with zipfile.ZipFile(zip_path) as archive:
            members = archive.infolist()
            if len(members) > 10_000:
                raise MinerUError("MinerU result ZIP contains too many entries")
            declared_size = sum(member.file_size for member in members)
            if declared_size > max_extracted_bytes:
                raise MinerUError(
                    "MinerU result ZIP exceeds the configured extracted-size limit"
                )

            root = temporary_dir.resolve()
            for member in members:
                member_path = PurePosixPath(member.filename)
                if member_path.is_absolute() or ".." in member_path.parts:
                    raise MinerUError(
                        f"Unsafe path in MinerU result ZIP: {member.filename}"
                    )
                if stat.S_ISLNK(member.external_attr >> 16):
                    raise MinerUError(
                        f"Symlink is not allowed in MinerU result ZIP: {member.filename}"
                    )
                target = (root / Path(*member_path.parts)).resolve()
                if target != root and root not in target.parents:
                    raise MinerUError(
                        f"Unsafe extraction target in MinerU result ZIP: {member.filename}"
                    )
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue

                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, target.open("wb") as destination:
                    while chunk := source.read(1024 * 1024):
                        extracted_bytes += len(chunk)
                        if extracted_bytes > max_extracted_bytes:
                            raise MinerUError(
                                "MinerU result ZIP exceeded the extracted-size limit"
                            )
                        destination.write(chunk)

        markdown_path = _find_markdown(temporary_dir, source_stem)
        if markdown_path is None:
            raise MinerUError("MinerU result ZIP did not contain a Markdown artifact")
        relative_markdown = markdown_path.relative_to(temporary_dir)
        if result_dir.exists():
            existing = _find_markdown(result_dir, source_stem)
            if existing is not None:
                return existing
            raise MinerUError(
                f"Incomplete MinerU result directory already exists: {result_dir}"
            )
        temporary_dir.replace(result_dir)
        return result_dir / relative_markdown
    finally:
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir, ignore_errors=True)


class MinerUClient:
    """Submit, recover, and download official MinerU asynchronous parse tasks."""

    def __init__(
        self,
        settings: Settings,
        *,
        async_client: httpx.AsyncClient | None = None,
        sync_client: httpx.Client | None = None,
    ) -> None:
        """Configure the service boundary and optional test transports."""
        self.settings = settings
        self.base_url = settings.mineru_base_url.rstrip("/")
        timeout = httpx.Timeout(
            connect=settings.external_connect_timeout_seconds,
            read=settings.external_read_timeout_seconds,
            write=settings.external_write_timeout_seconds,
            pool=settings.external_connect_timeout_seconds,
        )
        self._async_client = async_client or httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=False,
            trust_env=settings.external_trust_env,
        )
        self._sync_client = sync_client or httpx.Client(
            timeout=timeout,
            follow_redirects=False,
            trust_env=settings.external_trust_env,
        )
        self._owns_async_client = async_client is None
        self._owns_sync_client = sync_client is None
        self._locks: dict[str, asyncio.Lock] = {}

    async def parse(self, request: ParseRequest) -> ParseResult:
        """Create or recover one MinerU task for an idempotent parse attempt."""
        lock = self._locks.setdefault(request.idempotency_key, asyncio.Lock())
        async with lock:
            return await self._parse_locked(request)

    async def _parse_locked(self, request: ParseRequest) -> ParseResult:
        attempt_dir = (
            self.settings.artifacts_dir
            / request.job_id
            / f"parse-attempt-{request.attempt}"
        )
        result_dir = attempt_dir / "result"
        markdown_path = await asyncio.to_thread(
            _find_markdown,
            result_dir,
            Path(request.source_pdf).stem,
        )
        if markdown_path is not None:
            return ParseResult(
                markdown_path=str(markdown_path),
                idempotency_key=request.idempotency_key,
                created=False,
            )

        manifest_path = attempt_dir / "mineru-task.json"
        manifest = await asyncio.to_thread(_read_manifest, manifest_path)
        if manifest is not None and manifest.idempotency_key != request.idempotency_key:
            raise MinerUError(
                f"MinerU task manifest belongs to another request: {manifest_path}"
            )
        if manifest is not None:
            manifest = manifest.model_copy(
                update={
                    "status_url": self._pin_to_base_url(manifest.status_url),
                    "result_url": self._pin_to_base_url(manifest.result_url),
                }
            )

        if manifest is None:
            manifest = await asyncio.to_thread(self._submit_task, request)
            await asyncio.to_thread(
                _atomic_write_json,
                manifest_path,
                manifest.model_dump(mode="json"),
            )

        await self._wait_for_task(manifest)
        zip_path = await asyncio.to_thread(self._download_result, manifest, attempt_dir)
        try:
            markdown_path = await asyncio.to_thread(
                _safe_extract_result,
                zip_path,
                result_dir,
                max_extracted_bytes=self.settings.mineru_max_extracted_bytes,
                source_stem=Path(request.source_pdf).stem,
            )
        finally:
            await asyncio.to_thread(zip_path.unlink, missing_ok=True)
        return ParseResult(
            markdown_path=str(markdown_path),
            idempotency_key=request.idempotency_key,
            created=True,
        )

    def _submit_task(self, request: ParseRequest) -> _TaskManifest:
        source_path = Path(request.source_pdf)
        mime_type = mimetypes.guess_type(source_path.name)[0] or "application/pdf"
        parse_method = self.settings.mineru_parse_method
        if request.attempt > 1 and parse_method in {"auto", "txt"}:
            parse_method = "ocr"
        form_data = {
            "backend": self.settings.mineru_backend,
            "parse_method": parse_method,
            "return_md": "true",
            "return_images": "true",
            "response_format_zip": "true",
        }
        try:
            with source_path.open("rb") as source:
                response = self._sync_client.post(
                    f"{self.base_url}/tasks",
                    data=form_data,
                    files={"files": (source_path.name, source, mime_type)},
                )
        except httpx.TimeoutException as exc:
            raise MinerUTimeout("Timed out while submitting a MinerU task") from exc
        except httpx.HTTPError as exc:
            raise MinerUError(f"Failed to submit a MinerU task: {exc}") from exc

        payload = require_json_object(
            response,
            service="MinerU task submission",
            error_type=MinerUError,
            expected_status=202,
        )
        task_id = payload.get("task_id")
        status_url = payload.get("status_url")
        result_url = payload.get("result_url")
        if not isinstance(task_id, str) or not task_id:
            raise MinerUError("MinerU returned an invalid task id")
        if not isinstance(status_url, str) or not status_url:
            raise MinerUError("MinerU returned an invalid task status URL")
        if not isinstance(result_url, str) or not result_url:
            raise MinerUError("MinerU returned an invalid task payload")
        return _TaskManifest(
            idempotency_key=request.idempotency_key,
            task_id=task_id,
            status_url=self._pin_to_base_url(status_url),
            result_url=self._pin_to_base_url(result_url),
        )

    def _pin_to_base_url(self, returned_url: str) -> str:
        """Keep server-provided task paths on the configured MinerU origin."""
        base = urlsplit(self.base_url)
        returned = urlsplit(returned_url)
        path = returned.path
        if not path.startswith("/"):
            base_path = base.path.rstrip("/")
            path = f"{base_path}/{path}"
        return urlunsplit((base.scheme, base.netloc, path, returned.query, ""))

    async def _wait_for_task(self, manifest: _TaskManifest) -> None:
        deadline = time.monotonic() + self.settings.mineru_task_timeout_seconds
        while time.monotonic() < deadline:
            try:
                response = await self._async_client.get(manifest.status_url)
            except httpx.TimeoutException:
                await asyncio.sleep(self.settings.mineru_poll_interval_seconds)
                continue
            except httpx.HTTPError as exc:
                raise MinerUError(
                    f"Failed to poll MinerU task {manifest.task_id}: {exc}"
                ) from exc

            payload = require_json_object(
                response,
                service=f"MinerU task {manifest.task_id}",
                error_type=MinerUError,
            )
            status_value = payload.get("status")
            if status_value == "completed":
                return
            if status_value not in {"pending", "processing", "queued"}:
                raise MinerUError(
                    f"MinerU task {manifest.task_id} failed with status "
                    f"{status_value!r}: {payload}"
                )
            await asyncio.sleep(self.settings.mineru_poll_interval_seconds)
        raise MinerUTimeout(
            f"MinerU task {manifest.task_id} did not finish within "
            f"{self.settings.mineru_task_timeout_seconds:g}s"
        )

    def _download_result(
        self,
        manifest: _TaskManifest,
        attempt_dir: Path,
    ) -> Path:
        attempt_dir.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix="mineru-result-",
            suffix=".zip",
            dir=str(attempt_dir),
        )
        os.close(descriptor)
        zip_path = Path(temporary_name)
        downloaded = 0
        try:
            with self._sync_client.stream("GET", manifest.result_url) as response:
                if response.status_code != 200:
                    response.read()
                    raise MinerUError(
                        f"MinerU result download returned HTTP {response.status_code}: "
                        f"{response_detail(response)}"
                    )
                declared_length = response.headers.get("content-length")
                if declared_length:
                    try:
                        declared_bytes = int(declared_length)
                    except ValueError as exc:
                        raise MinerUError(
                            "MinerU returned an invalid result content-length"
                        ) from exc
                    if declared_bytes > self.settings.mineru_max_result_bytes:
                        raise MinerUError(
                            "MinerU result exceeds the configured download limit"
                        )
                with zip_path.open("wb") as destination:
                    for chunk in response.iter_bytes():
                        downloaded += len(chunk)
                        if downloaded > self.settings.mineru_max_result_bytes:
                            raise MinerUError(
                                "MinerU result exceeded the configured download limit"
                            )
                        destination.write(chunk)
            if not zipfile.is_zipfile(zip_path):
                raise MinerUError("MinerU result was not a valid ZIP archive")
            return zip_path
        except httpx.TimeoutException as exc:
            raise MinerUTimeout(
                f"Timed out while downloading MinerU task {manifest.task_id}"
            ) from exc
        except httpx.HTTPError as exc:
            raise MinerUError(
                f"Failed to download MinerU task {manifest.task_id}: {exc}"
            ) from exc
        except Exception:
            zip_path.unlink(missing_ok=True)
            raise

    async def aclose(self) -> None:
        """Close owned HTTP connection pools."""
        if self._owns_async_client:
            await self._async_client.aclose()
        if self._owns_sync_client:
            await asyncio.to_thread(self._sync_client.close)
