# -*- coding: utf-8 -*-
"""
Cloud control hub: master publishes task-sync events, slaves long-poll receive.

Room isolation: sha256(api_key)[:16] + ":" + normalize(master_name)
Protocol: docs aligned with game-get CLOUD_CONTROL.md
"""
from __future__ import annotations

import hashlib
import secrets
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

PROTOCOL_VERSION = 1
VALID_ROLES = frozenset({"master", "slave"})
VALID_ACTIONS = frozenset({"accept", "complete", "path", "claim_activity"})

# Tunables (env-overridable via configure())
DEFAULT_SESSION_TTL_S = 90.0
DEFAULT_EVENT_TTL_S = 60.0
DEFAULT_MAX_EVENTS_PER_ROOM = 200
DEFAULT_MAX_POLL_WAIT_S = 60.0
DEFAULT_MIN_POLL_WAIT_S = 1.0
DEFAULT_PUBLISH_RATE_PER_MIN = 60
DEFAULT_MAX_MASTER_NAME_LEN = 64


def normalize_master_name(name: str) -> str:
    s = " ".join(str(name or "").strip().split())
    return s[:DEFAULT_MAX_MASTER_NAME_LEN]


def room_key(api_key: str, master_name: str) -> str:
    digest = hashlib.sha256(str(api_key or "").encode("utf-8")).hexdigest()[:16]
    return f"{digest}:{normalize_master_name(master_name)}"


def new_session_id() -> str:
    return uuid.uuid4().hex


@dataclass
class Session:
    session_id: str
    room: str
    api_key_fp: str
    master_name: str
    role: str
    client_id: str
    pid: int = 0
    cursor: int = 0
    created_at: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)

    def touch(self) -> None:
        self.last_seen = time.time()


@dataclass
class RoomEvent:
    seq: int
    msg_id: str
    publisher_client_id: str
    publisher_session_id: str
    event: dict
    ts: float = field(default_factory=time.time)

    def to_wire(self) -> dict:
        return {
            "msg_id": self.msg_id,
            "event": dict(self.event),
        }


@dataclass
class Room:
    key: str
    master_name: str
    api_key_fp: str
    seq: int = 0
    events: list[RoomEvent] = field(default_factory=list)
    sessions: dict[str, Session] = field(default_factory=dict)  # session_id -> Session
    recent_msg_ids: dict[str, float] = field(default_factory=dict)
    publish_times: list[float] = field(default_factory=list)
    cond: threading.Condition = field(default_factory=threading.Condition)


class CloudControlHub:
    """
    In-memory multi-tenant room fan-out.

    Thread-safe; suitable for ThreadingHTTPServer long-poll.
    """

    def __init__(
        self,
        *,
        session_ttl_s: float = DEFAULT_SESSION_TTL_S,
        event_ttl_s: float = DEFAULT_EVENT_TTL_S,
        max_events_per_room: int = DEFAULT_MAX_EVENTS_PER_ROOM,
        publish_rate_per_min: int = DEFAULT_PUBLISH_RATE_PER_MIN,
    ) -> None:
        self._lock = threading.RLock()
        self._rooms: dict[str, Room] = {}
        self._sessions: dict[str, Session] = {}  # session_id -> Session
        self.session_ttl_s = float(session_ttl_s)
        self.event_ttl_s = float(event_ttl_s)
        self.max_events_per_room = int(max_events_per_room)
        self.publish_rate_per_min = int(publish_rate_per_min)

    # ---- public API -------------------------------------------------

    def join(
        self,
        *,
        api_key: str,
        master_name: str,
        role: str,
        client_id: str,
        pid: int = 0,
    ) -> dict[str, Any]:
        name = normalize_master_name(master_name)
        role_s = str(role or "").strip().lower()
        if not name:
            return {"ok": False, "code": 400, "message": "master_name 不能为空"}
        if role_s not in VALID_ROLES:
            return {"ok": False, "code": 400, "message": "role 必须是 master 或 slave"}
        cid = str(client_id or "").strip() or f"anon-{uuid.uuid4().hex[:8]}"
        rkey = room_key(api_key, name)
        fp = rkey.split(":", 1)[0]

        with self._lock:
            self._gc_locked()
            room = self._rooms.get(rkey)
            if room is None:
                room = Room(key=rkey, master_name=name, api_key_fp=fp)
                self._rooms[rkey] = room

            # Reuse session for same client_id in room when possible.
            existing = None
            for sess in list(room.sessions.values()):
                if sess.client_id == cid:
                    existing = sess
                    break
            if existing is not None:
                existing.role = role_s
                existing.pid = int(pid or 0)
                existing.master_name = name
                existing.touch()
                self._sessions[existing.session_id] = existing
                # New join for slave: start from current seq (no backlog flood)
                if role_s == "slave":
                    existing.cursor = room.seq
                return {
                    "ok": True,
                    "session_id": existing.session_id,
                    "cursor": str(existing.cursor),
                    "room": name,
                }

            sid = new_session_id()
            sess = Session(
                session_id=sid,
                room=rkey,
                api_key_fp=fp,
                master_name=name,
                role=role_s,
                client_id=cid,
                pid=int(pid or 0),
                cursor=room.seq,
            )
            room.sessions[sid] = sess
            self._sessions[sid] = sess
            return {
                "ok": True,
                "session_id": sid,
                "cursor": str(sess.cursor),
                "room": name,
            }

    def leave(
        self,
        *,
        api_key: str,
        master_name: str = "",
        session_id: str = "",
        client_id: str = "",
        **_kwargs: Any,
    ) -> dict[str, Any]:
        with self._lock:
            sess = self._resolve_session_locked(
                api_key=api_key,
                master_name=master_name,
                session_id=session_id,
                client_id=client_id,
            )
            if sess is None:
                return {"ok": True, "removed": False}
            self._drop_session_locked(sess.session_id)
            return {"ok": True, "removed": True}

    def heartbeat(
        self,
        *,
        api_key: str,
        master_name: str = "",
        session_id: str = "",
        client_id: str = "",
        role: str = "",
        pid: int = 0,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        with self._lock:
            self._gc_locked()
            sess = self._resolve_session_locked(
                api_key=api_key,
                master_name=master_name,
                session_id=session_id,
                client_id=client_id,
            )
            if sess is None:
                return {"ok": False, "code": 404, "message": "session 不存在或已过期，请重新 join"}
            if role in VALID_ROLES:
                sess.role = str(role).strip().lower()
            if pid:
                sess.pid = int(pid)
            sess.touch()
            return {
                "ok": True,
                "session_id": sess.session_id,
                "role": sess.role,
                "cursor": str(sess.cursor),
            }

    def publish(
        self,
        *,
        api_key: str,
        master_name: str,
        event: dict | None,
        client_id: str = "",
        session_id: str = "",
        msg_id: str = "",
        **_kwargs: Any,
    ) -> dict[str, Any]:
        name = normalize_master_name(master_name)
        if not name:
            return {"ok": False, "code": 400, "message": "master_name 不能为空"}
        if not isinstance(event, dict):
            # allow envelope already flattened
            return {"ok": False, "code": 400, "message": "event 必须是对象"}

        action = str(event.get("action") or "").strip().lower()
        if action not in VALID_ACTIONS:
            return {"ok": False, "code": 400, "message": f"不支持的 action: {action or '-'}"}
        try:
            task_id = int(event.get("task_id") or 0) & 0xFFFFFFFF
        except (TypeError, ValueError):
            task_id = 0
        if action != "claim_activity" and not task_id:
            return {"ok": False, "code": 400, "message": "task_id 无效"}

        rkey = room_key(api_key, name)
        with self._lock:
            self._gc_locked()
            room = self._rooms.get(rkey)
            if room is None:
                room = Room(
                    key=rkey,
                    master_name=name,
                    api_key_fp=rkey.split(":", 1)[0],
                )
                self._rooms[rkey] = room

            sess = self._resolve_session_locked(
                api_key=api_key,
                master_name=name,
                session_id=session_id,
                client_id=client_id,
            )
            if sess is None:
                return {
                    "ok": False,
                    "code": 403,
                    "message": "未 join 或 session 已过期，主控请先 join",
                }
            if sess.role != "master":
                return {"ok": False, "code": 403, "message": "仅主控可 publish"}
            if sess.room != rkey:
                return {"ok": False, "code": 403, "message": "session 与 master_name 不匹配"}

            now = time.time()
            # rate limit
            room.publish_times = [t for t in room.publish_times if now - t < 60.0]
            if len(room.publish_times) >= self.publish_rate_per_min:
                return {"ok": False, "code": 429, "message": "发布过于频繁，请稍后再试"}

            mid = str(msg_id or event.get("msg_id") or "").strip()
            if not mid:
                mid = f"{sess.client_id}-{action}-{task_id}-{int(now * 1000)}"
            # dedupe
            room.recent_msg_ids = {
                k: ts for k, ts in room.recent_msg_ids.items() if now - ts <= self.event_ttl_s
            }
            if mid in room.recent_msg_ids:
                return {"ok": True, "delivered": 0, "deduped": True, "msg_id": mid}

            wire_event = dict(event)
            wire_event["action"] = action
            wire_event["task_id"] = task_id
            wire_event["msg_id"] = mid
            wire_event["origin"] = "cloud"
            if "ts" not in wire_event:
                wire_event["ts"] = now
            if "source_pid" not in wire_event:
                wire_event["source_pid"] = sess.pid

            room.seq += 1
            item = RoomEvent(
                seq=room.seq,
                msg_id=mid,
                publisher_client_id=sess.client_id,
                publisher_session_id=sess.session_id,
                event=wire_event,
                ts=now,
            )
            room.events.append(item)
            room.recent_msg_ids[mid] = now
            room.publish_times.append(now)
            self._trim_room_events_locked(room)
            sess.touch()

            # count active slaves that will see this (not publisher)
            delivered = 0
            for other in room.sessions.values():
                if other.role != "slave":
                    continue
                if other.session_id == sess.session_id:
                    continue
                if other.client_id == sess.client_id:
                    continue
                if now - other.last_seen > self.session_ttl_s:
                    continue
                delivered += 1

            with room.cond:
                room.cond.notify_all()

            return {
                "ok": True,
                "delivered": delivered,
                "msg_id": mid,
                "seq": room.seq,
            }

    def poll(
        self,
        *,
        api_key: str,
        master_name: str,
        client_id: str = "",
        session_id: str = "",
        cursor: str = "",
        wait_s: float = 25.0,
    ) -> dict[str, Any]:
        name = normalize_master_name(master_name)
        if not name:
            return {"ok": False, "code": 400, "message": "master_name 不能为空"}

        wait = float(wait_s or 25.0)
        wait = max(DEFAULT_MIN_POLL_WAIT_S, min(DEFAULT_MAX_POLL_WAIT_S, wait))

        with self._lock:
            self._gc_locked()
            sess = self._resolve_session_locked(
                api_key=api_key,
                master_name=name,
                session_id=session_id,
                client_id=client_id,
            )
            if sess is None:
                return {
                    "ok": False,
                    "code": 404,
                    "message": "session 不存在或已过期，副控请先 join",
                }
            if sess.role != "slave":
                return {"ok": False, "code": 403, "message": "仅副控可 poll"}
            room = self._rooms.get(sess.room)
            if room is None:
                return {"ok": False, "code": 404, "message": "房间不存在"}

            # cursor from query overrides if provided and valid
            try:
                if cursor is not None and str(cursor).strip() != "":
                    sess.cursor = max(0, int(str(cursor).strip()))
            except (TypeError, ValueError):
                pass
            sess.touch()
            start_cursor = int(sess.cursor)
            cond = room.cond

        deadline = time.time() + wait
        while True:
            with self._lock:
                room = self._rooms.get(sess.room)
                if room is None or sess.session_id not in self._sessions:
                    return {
                        "ok": False,
                        "code": 404,
                        "message": "session 不存在或已过期，副控请先 join",
                    }
                # refresh session ref
                sess = self._sessions[sess.session_id]
                if sess.role != "slave":
                    return {"ok": False, "code": 403, "message": "仅副控可 poll"}
                events, new_cursor = self._events_after_locked(room, sess, start_cursor)
                sess.touch()
                if events:
                    sess.cursor = new_cursor
                    return {
                        "ok": True,
                        "cursor": str(new_cursor),
                        "events": events,
                    }
                remaining = deadline - time.time()
                if remaining <= 0:
                    return {
                        "ok": True,
                        "cursor": str(sess.cursor),
                        "events": [],
                    }
                cond = room.cond
                wait_chunk = min(1.0, remaining)

            # wait outside of hub lock but under room.cond (cond has its own lock)
            with cond:
                cond.wait(timeout=wait_chunk)

    def stats(self) -> dict[str, Any]:
        with self._lock:
            self._gc_locked()
            return {
                "rooms": len(self._rooms),
                "sessions": len(self._sessions),
            }

    # ---- internals --------------------------------------------------

    def _events_after_locked(
        self, room: Room, sess: Session, after_seq: int
    ) -> tuple[list[dict], int]:
        out: list[dict] = []
        max_seq = int(after_seq)
        now = time.time()
        for item in room.events:
            if item.seq <= after_seq:
                continue
            if now - item.ts > self.event_ttl_s:
                continue
            # never deliver publisher's own events to same client
            if item.publisher_client_id and item.publisher_client_id == sess.client_id:
                max_seq = max(max_seq, item.seq)
                continue
            if item.publisher_session_id == sess.session_id:
                max_seq = max(max_seq, item.seq)
                continue
            out.append(item.to_wire())
            max_seq = max(max_seq, item.seq)
        return out, max_seq

    def _resolve_session_locked(
        self,
        *,
        api_key: str,
        master_name: str = "",
        session_id: str = "",
        client_id: str = "",
    ) -> Session | None:
        sid = str(session_id or "").strip()
        if sid:
            sess = self._sessions.get(sid)
            if sess is None:
                return None
            # verify key fingerprint
            expect_fp = hashlib.sha256(str(api_key or "").encode("utf-8")).hexdigest()[:16]
            if sess.api_key_fp != expect_fp:
                return None
            if master_name:
                if normalize_master_name(master_name) != sess.master_name:
                    return None
            if time.time() - sess.last_seen > self.session_ttl_s:
                self._drop_session_locked(sid)
                return None
            return sess

        cid = str(client_id or "").strip()
        name = normalize_master_name(master_name)
        if not cid or not name:
            return None
        rkey = room_key(api_key, name)
        room = self._rooms.get(rkey)
        if room is None:
            return None
        for sess in room.sessions.values():
            if sess.client_id == cid:
                if time.time() - sess.last_seen > self.session_ttl_s:
                    self._drop_session_locked(sess.session_id)
                    return None
                return sess
        return None

    def _drop_session_locked(self, session_id: str) -> None:
        sess = self._sessions.pop(session_id, None)
        if sess is None:
            return
        room = self._rooms.get(sess.room)
        if room is not None:
            room.sessions.pop(session_id, None)
            if not room.sessions and not room.events:
                self._rooms.pop(sess.room, None)

    def _trim_room_events_locked(self, room: Room) -> None:
        now = time.time()
        room.events = [
            e for e in room.events if now - e.ts <= self.event_ttl_s
        ]
        if len(room.events) > self.max_events_per_room:
            room.events = room.events[-self.max_events_per_room :]

    def _gc_locked(self) -> None:
        now = time.time()
        expired = [
            sid
            for sid, sess in self._sessions.items()
            if now - sess.last_seen > self.session_ttl_s
        ]
        for sid in expired:
            self._drop_session_locked(sid)
        dead_rooms = []
        for rkey, room in self._rooms.items():
            self._trim_room_events_locked(room)
            if not room.sessions and not room.events:
                dead_rooms.append(rkey)
        for rkey in dead_rooms:
            self._rooms.pop(rkey, None)


_HUB: CloudControlHub | None = None
_HUB_LOCK = threading.Lock()


def get_cloud_control_hub() -> CloudControlHub:
    global _HUB
    with _HUB_LOCK:
        if _HUB is None:
            _HUB = CloudControlHub()
        return _HUB


def reset_cloud_control_hub_for_tests() -> CloudControlHub:
    global _HUB
    with _HUB_LOCK:
        _HUB = CloudControlHub()
        return _HUB
