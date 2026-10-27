"""Pemantau internet STT (26 Okt 2026): rute default OS + ping MAI, offline -> langsung whisper."""

import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from brone_talk.config import MaiConfig, SttConfig
from brone_talk.stt import transcriber as tr
from test_app_flow import FakeTelemetry

ROUTE_HEAD = "Iface\tDestination\tGateway\tFlags\tRefCnt\tUse\tMetric\tMask\tMTU\tWindow\tIRTT\n"


class FakePing:
    def __init__(self, ok):
        self.ok = ok

    def wait(self, timeout):
        return self.ok


class FakeWs:
    def __init__(self, ok=True):
        self.ok = ok
        self.closed = False

    def ping(self):
        return FakePing(self.ok)

    def close(self):
        self.closed = True


def make():
    t = tr.Transcriber(SttConfig(backend="mai"), MaiConfig(api_key="kunci"), FakeTelemetry())
    t._transcribe_whisper = lambda path: "whisper"
    return t


class DefaultRouteTest(unittest.TestCase):
    def write(self, body):
        handle = tempfile.NamedTemporaryFile("w", suffix=".route", delete=False)
        handle.write(ROUTE_HEAD + body)
        handle.close()
        self.addCleanup(Path(handle.name).unlink)
        return handle.name

    def test_default_route_present(self):
        path = self.write("wlo1\t00000000\t0164A8C0\t0003\t0\t0\t600\t00000000\t0\t0\t0\n")
        self.assertTrue(tr.has_default_route(path))

    def test_only_local_routes(self):
        path = self.write("wlo1\t0064A8C0\t00000000\t0001\t0\t0\t600\t00FFFFFF\t0\t0\t0\n")
        self.assertFalse(tr.has_default_route(path))

    def test_no_proc_counts_as_online(self):
        self.assertTrue(tr.has_default_route("/tidak/ada/route"))


class OfflineSkipTest(unittest.TestCase):
    def test_offline_goes_straight_to_whisper(self):
        t = make()
        t.cloud_online = False
        t._transcribe_mai = mock.Mock(side_effect=AssertionError("MAI tidak boleh dipanggil"))
        self.assertEqual(t._transcribe_mai_or_whisper(Path("x.wav")), "whisper")
        self.assertEqual(t.last_engine, "whisper")

    def test_online_uses_mai(self):
        t = make()
        t._transcribe_mai = lambda path: "UB di mana"
        self.assertEqual(t._transcribe_mai_or_whisper(Path("x.wav")), "UB di mana")
        self.assertEqual(t.last_engine, "mai")


class ProbeTest(unittest.TestCase):
    def test_mai_total_deadline_falls_back_to_whisper(self):
        t = make()
        t.mai_timeout_s = 0.2

        def slow(path):
            time.sleep(1.0)  # mis. menunggu kunci probe + kirim di jaringan lambat
            return "terlambat"

        t._transcribe_mai = slow
        started = time.monotonic()
        self.assertEqual(t._transcribe_mai_or_whisper(Path("x.wav")), "whisper")
        self.assertLess(time.monotonic() - started, 0.6)
        self.assertGreater(t.mai_skip_until, time.monotonic())

    def test_no_route_marks_offline(self):
        t = make()
        with mock.patch.object(tr, "has_default_route", return_value=False):
            t._probe_cloud()
        self.assertFalse(t.cloud_online)
        self.assertIn(("stt", "cloud_offline"), t.telemetry.stages)

    def test_ping_ok_recovers_and_ends_cooldown(self):
        t = make()
        t.cloud_online = False
        t.mai_skip_until = time.monotonic() + 30
        t._vl_ws, t._vl_opened = FakeWs(ok=True), time.monotonic()
        with mock.patch.object(tr, "has_default_route", return_value=True):
            t._probe_cloud()
        self.assertTrue(t.cloud_online)
        self.assertEqual(t.mai_skip_until, 0.0)
        self.assertIn(("stt", "cloud_online"), t.telemetry.stages)

    def test_dead_ping_reopens_and_open_failure_is_offline(self):
        t = make()
        ws = FakeWs(ok=False)
        t._vl_ws, t._vl_opened = ws, time.monotonic()
        t._voicelive_open = mock.Mock(side_effect=OSError("Temporary failure in name resolution"))
        with mock.patch.object(tr, "has_default_route", return_value=True):
            t._probe_cloud()
        t._voicelive_open.assert_called_once()
        self.assertFalse(t.cloud_online)

    def test_auth_cooldown_is_not_probed(self):
        t = make()
        t._mai_skip_auth = True
        t.mai_skip_until = time.monotonic() + 600
        t._voicelive_open = mock.Mock()
        with mock.patch.object(tr, "has_default_route", return_value=True):
            t._probe_cloud()
        t._voicelive_open.assert_not_called()
        self.assertGreater(t.mai_skip_until, time.monotonic())

    def test_busy_lock_skips_probe(self):
        t = make()
        t._voicelive_open = mock.Mock()
        with t._vl_lock, mock.patch.object(tr, "has_default_route", return_value=True):
            t._probe_cloud()
        t._voicelive_open.assert_not_called()
        self.assertTrue(t.cloud_online)


if __name__ == "__main__":
    unittest.main()
