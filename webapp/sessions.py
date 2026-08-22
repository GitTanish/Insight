from __future__ import annotations

import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional

from insight.domain.dataset import DatasetProfile
from insight.domain.visualization import AnalysisResponse


@dataclass
class WebSession:
    sid: str
    data_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    uploaded_name: Optional[str] = None
    content: Optional[bytes] = None
    profile: Optional[DatasetProfile] = None
    findings: Optional[list[dict]] = None
    model_id: Optional[str] = None
    temperature: Optional[float] = None
    turns: list[dict] = field(default_factory=list)
    analysis_state: Optional[dict] = None
    multi_content: dict[str, bytes] = field(default_factory=dict)

    def reset_data(self) -> None:
        self.data_id = uuid.uuid4().hex[:12]
        self.uploaded_name = None
        self.content = None
        self.profile = None
        self.findings = None
        self.turns = []
        self.analysis_state = None
        self.multi_content = {}

    @property
    def has_dataset(self) -> bool:
        return self.content is not None


class SessionStore:
    def __init__(self, max_sessions: int = 64):
        self._sessions: "OrderedDict[str, WebSession]" = OrderedDict()
        self._lock = threading.Lock()
        self._max_sessions = max_sessions

    def get(self, sid: str) -> WebSession | None:
        with self._lock:
            return self._sessions.get(sid)

    def get_or_create(self, sid: str) -> WebSession:
        with self._lock:
            session = self._sessions.get(sid)
            if session is None:
                while len(self._sessions) >= self._max_sessions:
                    _, oldest = self._sessions.popitem(last=False)
                session = WebSession(sid=sid)
                self._sessions[sid] = session
            else:
                self._sessions.move_to_end(sid)
            return session

    def drop(self, sid: str) -> None:
        with self._lock:
            self._sessions.pop(sid, None)


STORE = SessionStore()
