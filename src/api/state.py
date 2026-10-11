"""Lazily-loaded application singletons. Missing artifacts never crash startup; endpoints report 503 with guidance."""
from __future__ import annotations

import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

from ..agents.orchestrator import STAGES, AdvisoryOrchestrator
from ..models.yield_model import ModelUnavailable, YieldPredictor
from ..rag.retriever import HybridRetriever, get_retriever
from ..rag.store import CaseStore, DatasetUnavailable, get_store
from ..schemas.advisory import AdvisoryRequest
from ..services import db

log = logging.getLogger("terramind.state")


class AppState:
    def __init__(self):
        self._lock = threading.Lock()
        self._orch: AdvisoryOrchestrator | None = None
        self._predictor: YieldPredictor | None = None
        self._predictor_err: str | None = None
        self.jobs: dict[str, dict] = {}
        self.executor = ThreadPoolExecutor(max_workers=3)

    # --- components
    def store(self) -> CaseStore:
        return get_store()                      # raises DatasetUnavailable

    def retriever(self) -> HybridRetriever:
        return get_retriever(self.store())

    def predictor(self) -> YieldPredictor:
        if self._predictor is None:
            try:
                self._predictor = YieldPredictor()
                self._predictor_err = None
            except ModelUnavailable as e:
                self._predictor_err = str(e)
                raise
        return self._predictor

    def predictor_or_none(self) -> YieldPredictor | None:
        try:
            return self.predictor()
        except ModelUnavailable:
            return None

    def orchestrator(self) -> AdvisoryOrchestrator:
        with self._lock:
            if self._orch is None:
                self._orch = AdvisoryOrchestrator(self.store(), self.retriever(), self.predictor_or_none())
            return self._orch

    # --- advisory jobs (real stage progress, polled by the UI)
    def start_advisory_job(self, req: AdvisoryRequest) -> dict:
        orch = self.orchestrator()
        job_id, qid = uuid.uuid4().hex[:12], db.new_id("q")
        job = {"job_id": job_id, "query_id": qid, "status": "running", "created": time.time(),
               "stages": [{"key": k, "label": l, "status": "pending", "duration_ms": None, "message": None} for k, l in STAGES],
               "agents": [], "result": None, "error": None}
        self.jobs[job_id] = job
        if len(self.jobs) > 200:
            for k in sorted(self.jobs, key=lambda x: self.jobs[x]["created"])[:50]:
                self.jobs.pop(k, None)

        def progress(stages, agents):
            job["stages"], job["agents"] = stages, agents

        def work():
            try:
                res = orch.run(req, query_id=qid, on_progress=progress)
                job["result"] = res.model_dump()
                job["status"] = "completed"
            except Exception as e:  # noqa: BLE001
                log.exception("advisory job failed")
                job["status"], job["error"] = "failed", f"{type(e).__name__}: {e}"

        self.executor.submit(work)
        return {"job_id": job_id, "query_id": qid, "status": "running"}


state = AppState()
