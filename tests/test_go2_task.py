"""Direct regression for the complete-task stop-duration boundary."""
import sys
from pathlib import Path
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from go2_frozen.task import TerminalHold


class StopDuration(unittest.TestCase):
    def test_fifty_samples_do_not_span_one_second(self):
        timer = TerminalHold()
        for i in range(50):
            duration = timer.observe(7+i*.02, True)
        self.assertAlmostEqual(duration, .98)
        self.assertLess(duration, 1.)
        self.assertAlmostEqual(timer.observe(8., True), 1.)
        self.assertEqual(timer.observe(8.02, False), 0.)
        self.assertEqual(timer.observe(8.04, True), 0.)


if __name__ == "__main__":
    unittest.main()
