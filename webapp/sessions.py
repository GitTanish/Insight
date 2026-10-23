from __future__ import annotations

import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from insight.domain.dataset import DatasetProfile
from insight.domain.visualization import AnalysisResponse
from insight.settings import get_settings


@dataclass
class WebSession:
    sid: str
    data_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    uploaded_name: Optional[str] = None
    primary_table: Optional[str] = None
    content: Optional[bytes] = None
    frames: dict[str, pd.DataFrame] = field(default_factory=dict)
    profile: Optional[DatasetProfile] = None
    findings: Optional[list[dict]] = None
    model_id: Optional[str] = None
    temperature: Optional[float] = None
    turns: list[dict] = field(default_factory=list)
    analysis_state: Optional[dict] = None
    multi_content: dict[str, bytes] = field(default_factory=dict)
    # "ready" | "polishing" | "failed": LLM briefing polish runs in the
    # background so the upload response is not blocked by an LLM round-trip.
    findings_status: str = "ready"
    last_seen: float = field(default_factory=time.time)
    # Incremented whenever the dataset is replaced. Background work captures
    # it and refuses to write if a newer upload has landed meanwhile.
    generation: int = 0
    _state_lock: threading.RLock = field(
        default_factory=threading.RLock, repr=False, compare=False
    )
    _active_queries: int = field(default=0, repr=False, compare=False)
    _active_uploads: int = field(default=0, repr=False, compare=False)

    def reset_data(self) -> None:
        with self._state_lock:
            self.generation += 1
            self.data_id = uuid.uuid4().hex[:12]
            self.uploaded_name = None
            self.primary_table = None
            self.content = None
            self.frames = {}
            self.profile = None
            self.findings = None
            self.findings_status = "ready"
            self.turns = []
            self.analysis_state = None
            self.multi_content = {}

    @property
    def has_dataset(self) -> bool:
        with self._state_lock:
            return self.content is not None or bool(self.frames)

    @property
    def busy(self) -> bool:
        with self._state_lock:
            return self._active_queries > 0 or self._active_uploads > 0

    @property
    def active_queries(self) -> int:
        with self._state_lock:
            return self._active_queries

    @property
    def active_uploads(self) -> int:
        with self._state_lock:
            return self._active_uploads

    def begin_query(self) -> bool:
        """Reserve the session for one query; False if another op is running."""
        with self._state_lock:
            if self._active_queries or self._active_uploads:
                return False
            self._active_queries += 1
            return True

    def end_query(self) -> None:
        with self._state_lock:
            if self._active_queries:
                self._active_queries -= 1

    def begin_upload(self) -> bool:
        with self._state_lock:
            if self._active_queries or self._active_uploads:
                return False
            self._active_uploads += 1
            return True

    def end_upload(self) -> None:
        with self._state_lock:
            if self._active_uploads:
                self._active_uploads -= 1

    def replace_dataset(
        self,
        *,
        content: bytes,
        frames: dict[str, pd.DataFrame],
        primary_table: str,
        uploaded_name: str,
        profile: DatasetProfile,
        findings: list[dict],
        multi_content: dict[str, bytes],
    ) -> tuple[str, int]:
        """Atomically swap in a parsed upload; returns (old_data_id, generation)."""
        with self._state_lock:
            old_data_id = self.data_id
            self.reset_data()
            self.content = content
            self.frames = frames
            self.primary_table = primary_table
            self.uploaded_name = uploaded_name
            self.profile = profile
            self.findings = findings
            self.multi_content = multi_content
            return old_data_id, self.generation

    def dataset_snapshot(self) -> dict:
        """Consistent view of the current dataset for an in-flight query."""
        with self._state_lock:
            return {
                "data_id": self.data_id,
                "generation": self.generation,
                "content": self.content,
                "primary_table": self.primary_table,
                "frames": dict(self.frames),
                "multi_content": dict(self.multi_content),
                "profile": self.profile,
                "turns": list(self.turns),
                "analysis_state": self.analysis_state,
            }

    def add_turn(
        self,
        *,
        generation: int,
        data_id: str,
        question: str,
        response: AnalysisResponse,
        analysis_state: dict | None,
        max_turns: int = 25,
    ) -> bool:
        """Append a turn only if it belongs to the current dataset generation."""
        with self._state_lock:
            if generation != self.generation or data_id != self.data_id:
                return False
            self.turns.append({"question": question, "response": response})
            if analysis_state is not None:
                self.analysis_state = analysis_state
            if max_turns > 0 and len(self.turns) > max_turns:
                del self.turns[: len(self.turns) - max_turns]
            return True

    def turns_snapshot(self) -> list[dict]:
        with self._state_lock:
            return list(self.turns)

    def findings_snapshot(self) -> list[dict]:
        with self._state_lock:
            return list(self.findings or [])

    def polish_context(self) -> tuple[int, str, list[dict]]:
        with self._state_lock:
            return self.generation, self.data_id, list(self.findings or [])

    def set_findings(
        self, findings: list[dict], *, generation: int, data_id: str
    ) -> bool:
        with self._state_lock:
            if generation != self.generation or data_id != self.data_id:
                return False
            self.findings = findings
            self.findings_status = "ready"
            return True

    def set_findings_status(
        self, status: str, *, generation: int, data_id: str
    ) -> bool:
        with self._state_lock:
            if generation != self.generation or data_id != self.data_id:
                return False
            self.findings_status = status
            return True


class SessionStore:
    """Bounded, optionally idle-expiring in-memory session store.

    Ordered LRU eviction caps memory; ``ttl_seconds`` prevents stale sessions
    (and their uploaded bytes/frames) from living for the process lifetime.
    """

    def __init__(
        self,
        max_sessions: int = 64,
        ttl_seconds: float | None = None,
        clock=time.time,
    ):
        self._sessions: "OrderedDict[str, WebSession]" = OrderedDict()
        self._lock = threading.Lock()
        self._max_sessions = max(1, int(max_sessions))
        self._ttl = ttl_seconds
        self._clock = clock

    def _expired(self, session: WebSession) -> bool:
        if self._ttl is None:
            return False
        return (self._clock() - session.last_seen) > self._ttl

    def _drop_expired_head(self) -> None:
        while self._sessions:
            sid, session = next(iter(self._sessions.items()))
            if not self._expired(session) or session.busy:
                break
            self._sessions.pop(sid, None)

    def _evict_oldest_inactive(self) -> bool:
        for sid, session in self._sessions.items():
            if not session.busy:
                self._sessions.pop(sid, None)
                return True
        return False

    def get(self, sid: str) -> WebSession | None:
        if not sid:
            return None
        with self._lock:
            session = self._sessions.get(sid)
            if session is None:
                return None
            if self._expired(session) and not session.busy:
                self._sessions.pop(sid, None)
                return None
            session.last_seen = self._clock()
            self._sessions.move_to_end(sid)
            return session

    def get_or_create(self, sid: str) -> WebSession:
        with self._lock:
            self._drop_expired_head()
            session = self._sessions.get(sid)
            if session is not None and self._expired(session) and not session.busy:
                self._sessions.pop(sid, None)
                session = None
            if session is None:
                # An in-flight upload/query keeps its session even when the
                # store is full: correctness beats the memory cap temporarily.
                while len(self._sessions) >= self._max_sessions:
                    if not self._evict_oldest_inactive():
                        break
                session = WebSession(sid=sid, last_seen=self._clock())
                self._sessions[sid] = session
            else:
                session.last_seen = self._clock()
                self._sessions.move_to_end(sid)
            return session

    def drop(self, sid: str) -> None:
        with self._lock:
            self._sessions.pop(sid, None)

    def __len__(self) -> int:
        with self._lock:
            return len(self._sessions)


_SETTINGS = get_settings()
STORE = SessionStore(
    max_sessions=_SETTINGS.session_max,
    ttl_seconds=max(1, _SETTINGS.session_ttl_minutes) * 60,
)
