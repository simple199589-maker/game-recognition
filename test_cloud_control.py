# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from http import HTTPStatus
from pathlib import Path
from urllib.parse import urlparse, parse_qs
import urllib.request
import urllib.error
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from cloud_control import (  # noqa: E402
    CloudControlHub,
    normalize_master_name,
    reset_cloud_control_hub_for_tests,
    room_key,
)

from sameobject_training_web import (  # noqa: E402
    sign_license_request,
)
import hashlib
import hmac
import secrets as _secrets


class HubUnitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.hub = reset_cloud_control_hub_for_tests()

    def test_room_isolation_by_api_key(self) -> None:
        a = room_key("key-a", "主号")
        b = room_key("key-b", "主号")
        self.assertNotEqual(a, b)
        self.assertEqual(room_key("key-a", " 主  号 "), room_key("key-a", "主 号"))

    def test_master_publish_slave_receive(self) -> None:
        hub = self.hub
        m = hub.join(api_key="k1", master_name="房A", role="master", client_id="m1", pid=1)
        s = hub.join(api_key="k1", master_name="房A", role="slave", client_id="s1", pid=2)
        self.assertTrue(m["ok"] and s["ok"])
        pub = hub.publish(
            api_key="k1",
            master_name="房A",
            session_id=m["session_id"],
            client_id="m1",
            event={"action": "accept", "task_id": 10021, "source_pid": 1, "name": "每日"},
        )
        self.assertTrue(pub["ok"], pub)
        self.assertGreaterEqual(pub["delivered"], 1)
        polled = hub.poll(
            api_key="k1",
            master_name="房A",
            session_id=s["session_id"],
            client_id="s1",
            wait_s=1,
        )
        self.assertTrue(polled["ok"], polled)
        self.assertEqual(len(polled["events"]), 1)
        self.assertEqual(polled["events"][0]["event"]["action"], "accept")
        self.assertEqual(polled["events"][0]["event"]["task_id"], 10021)

        # second poll empty
        polled2 = hub.poll(
            api_key="k1",
            master_name="房A",
            session_id=s["session_id"],
            client_id="s1",
            wait_s=1,
        )
        self.assertEqual(polled2["events"], [])

    def test_slave_cannot_publish(self) -> None:
        hub = self.hub
        s = hub.join(api_key="k1", master_name="房", role="slave", client_id="s1")
        pub = hub.publish(
            api_key="k1",
            master_name="房",
            session_id=s["session_id"],
            client_id="s1",
            event={"action": "accept", "task_id": 1, "source_pid": 1},
        )
        self.assertFalse(pub["ok"])
        self.assertEqual(pub.get("code"), 403)

    def test_different_keys_do_not_share_room(self) -> None:
        hub = self.hub
        m = hub.join(api_key="k1", master_name="同名", role="master", client_id="m1")
        s = hub.join(api_key="k2", master_name="同名", role="slave", client_id="s2")
        hub.publish(
            api_key="k1",
            master_name="同名",
            session_id=m["session_id"],
            client_id="m1",
            event={"action": "complete", "task_id": 9, "source_pid": 1},
        )
        polled = hub.poll(
            api_key="k2",
            master_name="同名",
            session_id=s["session_id"],
            client_id="s2",
            wait_s=1,
        )
        self.assertEqual(polled["events"], [])

    def test_msg_id_dedupe(self) -> None:
        hub = self.hub
        m = hub.join(api_key="k1", master_name="房", role="master", client_id="m1")
        s = hub.join(api_key="k1", master_name="房", role="slave", client_id="s1")
        body = {
            "action": "path",
            "task_id": 3,
            "source_pid": 1,
            "portal_kind": "dungeon_upper",
            "msg_id": "dup-1",
        }
        p1 = hub.publish(
            api_key="k1",
            master_name="房",
            session_id=m["session_id"],
            client_id="m1",
            msg_id="dup-1",
            event=body,
        )
        p2 = hub.publish(
            api_key="k1",
            master_name="房",
            session_id=m["session_id"],
            client_id="m1",
            msg_id="dup-1",
            event=body,
        )
        self.assertTrue(p1["ok"])
        self.assertTrue(p2.get("deduped"))
        polled = hub.poll(
            api_key="k1",
            master_name="房",
            session_id=s["session_id"],
            client_id="s1",
            wait_s=1,
        )
        self.assertEqual(len(polled["events"]), 1)

    def test_claim_activity_task_id_zero(self) -> None:
        hub = self.hub
        m = hub.join(api_key="k1", master_name="房", role="master", client_id="m1")
        s = hub.join(api_key="k1", master_name="房", role="slave", client_id="s1")
        pub = hub.publish(
            api_key="k1",
            master_name="房",
            session_id=m["session_id"],
            client_id="m1",
            event={"action": "claim_activity", "task_id": 0, "points": 35, "name": "回城"},
        )
        self.assertTrue(pub["ok"], pub)
        polled = hub.poll(
            api_key="k1",
            master_name="房",
            session_id=s["session_id"],
            client_id="s1",
            wait_s=1,
        )
        self.assertEqual(polled["events"][0]["event"]["action"], "claim_activity")
        self.assertEqual(polled["events"][0]["event"]["points"], 35)

    def test_map_fly_and_unknown_action_passthrough(self) -> None:
        """任意非空 action 透传：不维护白名单，event 字段原样到达副控。"""
        hub = self.hub
        m = hub.join(api_key="k1", master_name="房", role="master", client_id="m1", pid=11)
        s = hub.join(api_key="k1", master_name="房", role="slave", client_id="s1", pid=22)
        for name in ("fuzhou", "shimen", "death"):
            pub = hub.publish(
                api_key="k1",
                master_name="房",
                session_id=m["session_id"],
                client_id="m1",
                event={"action": "map_fly", "task_id": 0, "name": name, "extra": {"x": 1}},
            )
            self.assertTrue(pub["ok"], pub)
            polled = hub.poll(
                api_key="k1",
                master_name="房",
                session_id=s["session_id"],
                client_id="s1",
                wait_s=1,
            )
            self.assertEqual(len(polled["events"]), 1, polled)
            ev = polled["events"][0]["event"]
            self.assertEqual(ev["action"], "map_fly")
            self.assertEqual(ev["name"], name)
            self.assertEqual(ev["extra"], {"x": 1})
            self.assertEqual(ev["origin"], "cloud")
            self.assertEqual(ev["task_id"], 0)

        # 任意新 action 同样透传
        pub2 = hub.publish(
            api_key="k1",
            master_name="房",
            session_id=m["session_id"],
            client_id="m1",
            event={"action": "custom_sync", "foo": "bar", "task_id": 42},
        )
        self.assertTrue(pub2["ok"], pub2)
        polled2 = hub.poll(
            api_key="k1",
            master_name="房",
            session_id=s["session_id"],
            client_id="s1",
            wait_s=1,
        )
        self.assertEqual(polled2["events"][0]["event"]["action"], "custom_sync")
        self.assertEqual(polled2["events"][0]["event"]["foo"], "bar")
        self.assertEqual(polled2["events"][0]["event"]["task_id"], 42)

        # 空 action 仍拒绝
        bad = hub.publish(
            api_key="k1",
            master_name="房",
            session_id=m["session_id"],
            client_id="m1",
            event={"action": "", "task_id": 1},
        )
        self.assertFalse(bad["ok"])
        self.assertEqual(bad.get("code"), 400)

    def test_normalize_master_name(self) -> None:
        self.assertEqual(normalize_master_name("  a  b  "), "a b")



class _CloudOnlyHandler(BaseHTTPRequestHandler):
    api_key = "test-key"
    sign_secret = "test-license-secret"
    hub: CloudControlHub

    def log_message(self, fmt, *args):
        return

    def send_json(self, status: HTTPStatus, payload: dict) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def authorized(self) -> bool:
        return (self.headers.get("X-API-Key") or "") == self.api_key

    def read_raw(self) -> bytes:
        n = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(n) if n else b""

    def require_sign(self, method: str, path: str, raw: bytes) -> bool:
        ts = (self.headers.get("X-Timestamp") or "").strip()
        nonce = (self.headers.get("X-Nonce") or "").strip()
        sig = (self.headers.get("X-Signature") or "").strip().lower()
        if not ts or not nonce or not sig:
            self.send_json(HTTPStatus.UNAUTHORIZED, {"ok": False, "code": 401, "message": "missing sign headers"})
            return False
        expected = sign_license_request(self.sign_secret, method, path, ts, nonce, raw or b"")
        if not hmac.compare_digest(expected, sig):
            self.send_json(HTTPStatus.UNAUTHORIZED, {"ok": False, "code": 401, "message": "bad signature"})
            return False
        # simple in-memory nonce set on hub object
        used = getattr(self.hub, "_test_nonces", None)
        if used is None:
            used = set()
            self.hub._test_nonces = used
        if nonce in used:
            self.send_json(HTTPStatus.CONFLICT, {"ok": False, "code": 409, "message": "nonce replay"})
            return False
        used.add(nonce)
        return True

    def do_POST(self):  # noqa: N802
        path = urlparse(self.path).path
        raw = self.read_raw()
        if not self.require_sign("POST", path, raw):
            return
        if not self.authorized():
            self.send_json(HTTPStatus.UNAUTHORIZED, {"ok": False, "code": 401})
            return
        body = json.loads(raw.decode("utf-8") or "{}") if raw else {}
        hub = self.hub
        key = self.api_key
        if path.endswith("/join"):
            r = hub.join(
                api_key=key,
                master_name=body.get("master_name", ""),
                role=body.get("role", ""),
                client_id=body.get("client_id", ""),
                pid=int(body.get("pid") or 0),
            )
        elif path.endswith("/publish"):
            r = hub.publish(
                api_key=key,
                master_name=body.get("master_name", ""),
                event=body.get("event"),
                client_id=body.get("client_id", ""),
                session_id=body.get("session_id", ""),
                msg_id=body.get("msg_id", ""),
            )
        elif path.endswith("/leave"):
            r = hub.leave(
                api_key=key,
                master_name=body.get("master_name", ""),
                client_id=body.get("client_id", ""),
                session_id=body.get("session_id", ""),
            )
        else:
            r = {"ok": False, "code": 404}
        self.send_json(HTTPStatus.OK if r.get("ok") else HTTPStatus.BAD_REQUEST, r)

    def do_GET(self):  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        sign_path = path + (("?" + parsed.query) if parsed.query else "")
        if not self.require_sign("GET", sign_path, b""):
            return
        if not self.authorized():
            self.send_json(HTTPStatus.UNAUTHORIZED, {"ok": False, "code": 401})
            return
        if not path.endswith("/poll"):
            self.send_json(HTTPStatus.NOT_FOUND, {"ok": False})
            return
        qs = parse_qs(parsed.query)
        def q(n, d=""):
            return (qs.get(n) or [d])[0]
        r = self.hub.poll(
            api_key=self.api_key,
            master_name=q("master_name"),
            client_id=q("client_id"),
            session_id=q("session_id"),
            cursor=q("cursor"),
            wait_s=float(q("wait_s", "2")),
        )
        self.send_json(HTTPStatus.OK if r.get("ok") else HTTPStatus.BAD_REQUEST, r)


def _sign_headers(method: str, path: str, raw: bytes, secret: str) -> dict[str, str]:
    ts = str(int(time.time()))
    nonce = _secrets.token_hex(16)
    sig = sign_license_request(secret, method, path, ts, nonce, raw or b"")
    return {"X-Timestamp": ts, "X-Nonce": nonce, "X-Signature": sig}


class HttpBridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.hub = reset_cloud_control_hub_for_tests()
        _CloudOnlyHandler.hub = cls.hub
        cls.secret = _CloudOnlyHandler.sign_secret
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), _CloudOnlyHandler)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"
        cls.th = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.th.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def _json(self, method: str, path: str, body: dict | None = None, query: str = "") -> dict:
        data = None if body is None else json.dumps(body).encode("utf-8")
        q = query
        url = self.base + path + (("?" + q) if q else "")
        sign_path = path + (("?" + q) if q else "")
        headers = {
            "X-API-Key": "test-key",
            **_sign_headers(method, sign_path, data or b"", self.secret),
        }
        if body is not None:
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, method=method)
        for k, v in headers.items():
            req.add_header(k, v)
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def test_http_master_to_slave(self) -> None:
        name = f"http-room-{time.time_ns()}"
        m = self._json(
            "POST",
            "/api/cloud-control/join",
            {"master_name": name, "role": "master", "client_id": "hm", "pid": 11},
        )
        s = self._json(
            "POST",
            "/api/cloud-control/join",
            {"master_name": name, "role": "slave", "client_id": "hs", "pid": 22},
        )
        self.assertTrue(m["ok"] and s["ok"])
        pub = self._json(
            "POST",
            "/api/cloud-control/publish",
            {
                "master_name": name,
                "client_id": "hm",
                "session_id": m["session_id"],
                "msg_id": f"m-{time.time_ns()}",
                "event": {
                    "action": "accept",
                    "task_id": 777,
                    "source_pid": 11,
                    "name": "云任务",
                    "origin": "cloud",
                },
            },
        )
        self.assertTrue(pub["ok"], pub)
        q = (
            f"master_name={name}&client_id=hs&session_id={s['session_id']}"
            f"&cursor={s['cursor']}&wait_s=2"
        )
        polled = self._json("GET", "/api/cloud-control/poll", query=q)
        self.assertTrue(polled["ok"], polled)
        self.assertEqual(len(polled["events"]), 1)
        self.assertEqual(polled["events"][0]["event"]["task_id"], 777)

    def test_reject_unsigned(self) -> None:
        url = self.base + "/api/cloud-control/join"
        body = json.dumps({"master_name": "x", "role": "master", "client_id": "u"}).encode()
        req = urllib.request.Request(url, data=body, method="POST")
        req.add_header("X-API-Key", "test-key")
        req.add_header("Content-Type", "application/json")
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(req, timeout=5)
        self.assertEqual(ctx.exception.code, 401)


if __name__ == "__main__":
    unittest.main()
