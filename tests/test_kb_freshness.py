"""Peringatan data KB bertanggal (berlaku_hingga) yang sudah atau hampir kedaluwarsa."""

import unittest

from brone_talk.engine.util import expiry_report


class ExpiryReportTest(unittest.TestCase):
    def test_expired_soon_and_ignored(self):
        records = [
            {"id": "wisuda-lama", "berlaku_hingga": "2026-10-11"},
            {"id": "acara-dekat", "berlaku_hingga": "2026-11-01"},
            {"id": "biaya", "berlaku_hingga": "2027-06-30"},
            {"id": "rektor", "berlaku_hingga": None},
        ]
        expired, soon = expiry_report(records, "2026-10-26")
        self.assertEqual(expired, ["wisuda-lama"])
        self.assertEqual(soon, ["acara-dekat (2026-11-01)"])

    def test_due_today_is_still_valid(self):
        expired, soon = expiry_report([{"id": "x", "berlaku_hingga": "2026-10-26"}], "2026-10-26")
        self.assertEqual(expired, [])
        self.assertEqual(soon, ["x (2026-10-26)"])


if __name__ == "__main__":
    unittest.main()
