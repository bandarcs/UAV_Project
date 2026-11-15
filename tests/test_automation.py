"""Policy checks do not measure detector accuracy."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from uav_project import RateController, checked_video, class_mapping, select_profile, open_result


def profile(name="n.pt", size=640, ms=10, parameters=3, classes=("airplane", "bird"), ok=True):
    return dict(weights=name, imgsz=size, latency_ms=ms, parameters=parameters,
                supported_classes=list(classes), ok=ok)


class AutomationPolicyTests(unittest.TestCase):
    def test_player_launches_completed_local_file(self):
        with patch('uav_project.sys.platform', 'win32'), patch('uav_project.os.startfile', create=True) as launch:
            self.assertTrue(open_result(Path('annotated.mp4')))
            launch.assert_called_once_with(str(Path('annotated.mp4').resolve()))

    def test_player_failure_keeps_results_available(self):
        with patch('uav_project.sys.platform', 'win32'), patch('uav_project.os.startfile', create=True, side_effect=OSError('No player')), patch('builtins.print'):
            self.assertFalse(open_result(Path('annotated.mp4')))

    def test_missing_classes_are_not_invented(self):
        mapping = class_mapping({0: "person", 4: "airplane", 14: "bird"})
        self.assertEqual(mapping, {4: "airplane", 14: "bird"})

    def test_faster_model_cannot_silently_remove_supported_classes(self):
        four = tuple(sorted(("airplane", "bird", "helicopter", "drone")))
        picked, _, _ = select_profile([profile(ms=1), profile("custom.pt", ms=30, classes=four)], 30, False)
        self.assertEqual(set(picked["supported_classes"]), set(four))

    def test_unsupported_fast_model_is_excluded(self):
        picked, _, _ = select_profile([profile(ms=1, classes=()), profile("working.pt")], 30, False)
        self.assertEqual(picked["weights"], "working.pt")

    def test_failed_profile_is_never_selected(self):
        picked, _, _ = select_profile([profile("failed.pt", ms=1, ok=False), profile("ok.pt")], 30, False)
        self.assertEqual(picked["weights"], "ok.pt")

    def test_fastest_fallback_schedules_slow_hardware(self):
        picked, stride, _ = select_profile([profile(ms=100), profile("slow.pt", ms=400)], 30, False)
        self.assertEqual(picked["weights"], "n.pt")
        self.assertGreater(stride, 1)

    def test_budget_rejects_profile_too_slow_for_source(self):
        picked, _, _ = select_profile([profile("fast.pt", ms=10), profile("slow.pt", size=1280, ms=150)], 30, False)
        self.assertEqual(picked["weights"], "fast.pt")

    def test_quality_heuristic_changes_preferences_without_accuracy_claim(self):
        candidates = [profile("n.pt", size=1280, parameters=3), profile("m.pt", size=640, parameters=25)]
        detail, _, _ = select_profile(candidates, 30, False)
        difficult, _, _ = select_profile(candidates, 30, True)
        self.assertEqual(detail["weights"], "n.pt")
        self.assertEqual(difficult["weights"], "m.pt")

    def test_invalid_or_empty_trials_stop(self):
        for candidates in ([], [profile(ms=float("nan"))], [profile(ok=False)]):
            with self.assertRaises(ValueError):
                select_profile(candidates, 30, False)

    def test_high_rate_input_respects_analysis_rate_cap(self):
        _, stride, _ = select_profile([profile(ms=1)], 120, False)
        self.assertEqual(stride, 4)

    def test_recorded_video_processes_every_frame_and_can_choose_a_slower_validated_profile(self):
        fast = dict(profile("fast.pt"), validation_map50=0.2)
        accurate = dict(profile("accurate.pt", ms=100), validation_map50=0.6)
        recorded, stride, _ = select_profile([fast, accurate], 60, False, recorded=True)
        live, _, _ = select_profile([fast, accurate], 60, False)
        self.assertEqual((recorded["weights"], stride), ("accurate.pt", 1))
        self.assertEqual(live["weights"], "fast.pt")

    def test_validated_profile_beats_unmeasured_model_size(self):
        large = profile("large.pt", size=1280, parameters=50)
        measured = dict(profile("measured.pt"), validation_map50=0.4)
        picked, _, _ = select_profile([large, measured], 30, False)
        self.assertEqual(picked["weights"], "measured.pt")

    def test_scheduler_recovers_from_stride_two_to_one(self):
        controller = RateController(30, 2)
        self.assertIsNone(controller.update(10))
        self.assertEqual(controller.update(10), 1)

    def test_scheduler_increases_promptly_without_oscillation(self):
        controller = RateController(30, 1)
        self.assertEqual(controller.update(40), 2)
        self.assertIsNone(controller.update(10))
        self.assertIsNone(controller.update(40))
        self.assertIsNone(controller.update(10))
        self.assertEqual(controller.update(10), 1)

    def test_reject_camera_and_network_inputs(self):
        for source in ("0", "rtsp://camera/feed", "https://example.org/video.mp4"):
            with self.assertRaises(ValueError):
                checked_video(source)


if __name__ == "__main__":
    unittest.main()
