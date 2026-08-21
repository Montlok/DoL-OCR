# -*- coding: utf-8 -*-

"""StatusReporter + monitor HTTP API tests."""

from __future__ import annotations

import json
import threading
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer

import pytest

from Model.training.status import StatusReporter, read_history, read_status


class TestStatusReporter:
    def test_update_and_read(self, tmp_path):
        rep = StatusReporter(tmp_path, max_steps=100, run_metadata={"config_name": "tiny"})
        rep.update(5, {"loss": 1.25, "lr": 1e-4})

        status = read_status(tmp_path)
        assert status["step"] == 5
        assert status["state"] == "running"
        assert status["metrics"]["loss"] == 1.25
        assert status["progress"] == 0.05
        assert status["run"]["config_name"] == "tiny"

    def test_history_ring(self, tmp_path):
        rep = StatusReporter(tmp_path, history_keep=10)
        for step in range(50):
            rep.update(step, {"loss": float(step)})

        rows = read_history(tmp_path, last_n=100)
        assert len(rows) <= 20
        assert rows[-1]["step"] == 49

    def test_history_lines_resume(self, tmp_path):
        rep = StatusReporter(tmp_path, history_keep=10)
        for step in range(15):
            rep.update(step, {"loss": float(step)})

        # a resumed reporter must count pre-existing lines so the ring
        # trim threshold doesn't restart from zero
        rep2 = StatusReporter(tmp_path, history_keep=10)
        assert rep2._history_lines == 15
        for step in range(15, 30):
            rep2.update(step, {"loss": float(step)})

        rows = read_history(tmp_path, last_n=100)
        assert len(rows) <= 20
        assert rows[-1]["step"] == 29

    def test_nonfinite_metrics_become_null(self, tmp_path):
        rep = StatusReporter(tmp_path)
        rep.update(1, {"loss": float("nan"), "grad_norm": float("inf"), "lr": 0.1})

        raw = (tmp_path / "monitor" / "status.json").read_text(encoding="utf-8")
        status = json.loads(raw)  # strict: would fail on bare NaN/Infinity
        assert "NaN" not in raw and "Infinity" not in raw
        assert status["metrics"]["loss"] is None
        assert status["metrics"]["grad_norm"] is None
        assert status["metrics"]["lr"] == 0.1

    def test_finish_state(self, tmp_path):
        rep = StatusReporter(tmp_path)
        rep.finish(state="stopped", step=7)
        assert read_status(tmp_path)["state"] == "stopped"

    def test_control_roundtrip(self, tmp_path):
        rep = StatusReporter(tmp_path)
        assert rep.poll_control() == []

        StatusReporter.request(tmp_path, "save", "stop")
        StatusReporter.request(tmp_path, "save")  # dedup
        assert rep.poll_control() == ["save", "stop"]
        assert rep.poll_control() == []  # consumed

    def test_control_rejects_unknown(self, tmp_path):
        with pytest.raises(ValueError):
            StatusReporter.request(tmp_path, "reboot")

    def test_corrupt_control_ignored(self, tmp_path):
        rep = StatusReporter(tmp_path)
        rep.control_path.write_text("{not json", encoding="utf-8")
        assert rep.poll_control() == []

    def test_tensor_metrics_jsonable(self, tmp_path):
        torch = pytest.importorskip("torch")
        rep = StatusReporter(tmp_path)
        rep.update(1, {"loss": torch.tensor(2.5)})
        assert read_status(tmp_path)["metrics"]["loss"] == 2.5


class TestMonitorAPI:
    @pytest.fixture()
    def server(self, tmp_path):
        from scripts.rdt_monitor import _make_handler

        rep = StatusReporter(tmp_path, max_steps=10)
        rep.update(3, {"loss": 0.5})

        httpd = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(str(tmp_path)))
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        yield tmp_path, rep, httpd.server_address[1]
        httpd.shutdown()
        httpd.server_close()

    def _get(self, port, path):
        conn = HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("GET", path)
        resp = conn.getresponse()
        return resp.status, json.loads(resp.read())

    def _post(self, port, path, payload):
        conn = HTTPConnection("127.0.0.1", port, timeout=5)
        body = json.dumps(payload)
        conn.request("POST", path, body, {"Content-Type": "application/json"})
        resp = conn.getresponse()
        return resp.status, json.loads(resp.read())

    def test_health_status_history(self, server):
        _tmp, _rep, port = server
        assert self._get(port, "/api/health")[0] == 200

        code, status = self._get(port, "/api/status")
        assert code == 200 and status["step"] == 3

        code, hist = self._get(port, "/api/history?n=10")
        assert code == 200 and hist["history"][-1]["loss"] == 0.5

    def test_control_post_reaches_trainer(self, server):
        _tmp, rep, port = server
        code, body = self._post(port, "/api/control", {"command": "save"})
        assert code == 200 and body["queued"] == "save"
        assert rep.poll_control() == ["save"]

    def test_bad_inputs(self, server):
        _tmp, _rep, port = server
        assert self._post(port, "/api/control", {"command": "rm -rf"})[0] == 400
        assert self._get(port, "/api/history?n=zero")[0] == 400
        assert self._get(port, "/api/nope")[0] == 404
