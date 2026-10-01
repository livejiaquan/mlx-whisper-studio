from pathlib import Path
import os
import queue
import subprocess
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from app import MLXWhisperApp
from output_files import cleanup_outputs, prepare_outputs, publish_outputs


class OutputSafetyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def plan(self, formats=None):
        return prepare_outputs(str(self.root / "lecture.wav"), formats or ["txt"], str(self.root))

    def write_outputs(self, plan, names=None):
        for name in names or [f"lecture.{fmt}" for fmt in plan.formats]:
            (Path(plan.staging_dir) / name).write_text("new transcript", encoding="utf-8")

    def test_failed_single_format_preserves_existing_transcript(self):
        old = self.root / "lecture.txt"
        old.write_bytes(b"existing transcript")
        plan = self.plan()
        self.write_outputs(plan)
        cleanup_outputs(plan)
        self.assertEqual(old.read_bytes(), b"existing transcript")
        self.assertFalse(Path(plan.staging_dir).exists())

    def test_failed_multi_format_preserves_existing_files_and_translations(self):
        folder = self.root / "lecture_outputs"
        folder.mkdir()
        old_names = ["lecture.txt", "lecture.srt", "lecture_zh.txt", "notes.txt"]
        for name in old_names:
            (folder / name).write_bytes(b"old")
        plan = self.plan(["txt", "srt"])
        self.write_outputs(plan, ["lecture.txt", "lecture.srt", "lecture_zh.txt"])
        cleanup_outputs(plan)
        self.assertEqual(sorted(path.name for path in folder.iterdir()), sorted(old_names))
        self.assertTrue(all((folder / name).read_bytes() == b"old" for name in old_names))

    def test_success_publishes_all_formats_and_translation(self):
        plan = self.plan(["txt", "srt", "vtt", "json"])
        names = [f"lecture{suffix}.{fmt}" for suffix in ("", "_zh") for fmt in plan.formats]
        self.write_outputs(plan, names)
        published = publish_outputs(plan)
        cleanup_outputs(plan)
        self.assertEqual(sorted(Path(path).name for path in published), sorted(names))
        self.assertTrue(all(Path(path).read_text() == "new transcript" for path in published))

    def test_collisions_use_one_suffix_and_never_touch_old_files(self):
        plan = self.plan(["txt", "srt"])
        folder = Path(plan.output_dir)
        old = folder / "lecture_zh.txt"
        old.write_text("old translation")
        self.write_outputs(plan, ["lecture.txt", "lecture.srt", "lecture_zh.txt"])
        published = publish_outputs(plan)
        cleanup_outputs(plan)
        self.assertEqual(sorted(Path(path).name for path in published),
                         ["lecture_2.srt", "lecture_2.txt", "lecture_2_zh.txt"])
        self.assertEqual(old.read_text(), "old translation")
        self.assertFalse((folder / "lecture.txt").exists())

    def test_two_runs_with_same_basename_do_not_overwrite(self):
        first, second = self.plan(), self.plan()
        self.write_outputs(first)
        self.write_outputs(second)
        one = publish_outputs(first)
        two = publish_outputs(second)
        cleanup_outputs(first)
        cleanup_outputs(second)
        self.assertEqual(Path(one[0]).name, "lecture.txt")
        self.assertEqual(Path(two[0]).name, "lecture_2.txt")

    def test_missing_output_does_not_publish_or_remove_any_existing_file(self):
        plan = self.plan(["txt", "srt"])
        old = Path(plan.output_dir) / "lecture.txt"
        old.write_text("old")
        self.write_outputs(plan, ["lecture.txt"])
        with self.assertRaisesRegex(RuntimeError, "all requested"):
            publish_outputs(plan)
        cleanup_outputs(plan)
        self.assertEqual(old.read_text(), "old")
        self.assertEqual(list(old.parent.iterdir()), [old])

    def test_publish_failure_rolls_back_only_new_files(self):
        old = self.root / "lecture.txt"
        old.write_text("old")
        plan = self.plan()
        self.write_outputs(plan)
        with patch("output_files.shutil.copyfileobj", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                publish_outputs(plan)
        cleanup_outputs(plan)
        self.assertEqual(old.read_text(), "old")
        self.assertEqual(list(self.root.iterdir()), [old])

    def test_failed_new_multi_format_run_removes_empty_output_folder(self):
        plan = self.plan(["txt", "srt"])
        self.write_outputs(plan, ["lecture.txt"])
        cleanup_outputs(plan)
        self.assertFalse(Path(plan.output_dir).exists())

    def test_cleanup_keeps_another_writers_file_in_new_folder(self):
        plan = self.plan(["txt", "srt"])
        other = Path(plan.output_dir) / "notes.txt"
        other.write_text("keep")
        cleanup_outputs(plan)
        self.assertEqual(other.read_text(), "keep")

    def test_symlink_output_is_not_published(self):
        plan = self.plan()
        old = self.root / "outside.txt"
        old.write_text("old")
        (Path(plan.staging_dir) / "lecture.txt").symlink_to(old)
        with self.assertRaisesRegex(RuntimeError, "Unexpected"):
            publish_outputs(plan)
        cleanup_outputs(plan)
        self.assertEqual(old.read_text(), "old")


class QueueSafetyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.app = MLXWhisperApp.__new__(MLXWhisperApp)
        self.app.cancel_requested = False
        self.app.current_process = None
        self.app.current_plan = None
        self.app.log_queue = queue.Queue()
        self.app.queue_formats = ["txt"]
        self.app.queue_output_root = str(self.root)
        self.app.queue = [str(self.root / "lecture.wav")]
        self.app._post_ui = lambda callback: callback()
        self.app._update_current_file = lambda text: None
        self.app._log = lambda text: self.app.log_queue.put(text)
        self.statuses = []
        self.app._mark_item_status = lambda index, status: self.statuses.append((index, status))
        self.finished = []
        self.app._finish_queue = lambda: self.finished.append(True)

    def test_spawn_failure_preserves_old_file_and_finishes_queue(self):
        old = self.root / "lecture.txt"
        old.write_text("old")
        self.app._run_worker = lambda *_: (_ for _ in ()).throw(OSError("cannot start"))
        self.app._process_queue()
        self.assertEqual(old.read_text(), "old")
        self.assertEqual(self.finished, [True])
        self.assertEqual(self.statuses[-1], (0, "Failed"))
        self.assertEqual(list(self.root.iterdir()), [old])

    def test_retry_after_failure_produces_new_file_and_keeps_original(self):
        old = self.root / "lecture.txt"
        old.write_text("old")

        def fail(_file, staging):
            (Path(staging) / "lecture.txt").write_text("partial")
            return False

        self.app._run_worker = fail
        self.app._process_queue()
        self.assertEqual(list(self.root.iterdir()), [old])

        def succeed(_file, staging):
            (Path(staging) / "lecture.txt").write_text("complete")
            return True

        self.app._run_worker = succeed
        self.app._process_queue()
        self.assertEqual(old.read_text(), "old")
        self.assertEqual((self.root / "lecture_2.txt").read_text(), "complete")
        self.assertEqual(self.statuses[-1], (0, "Done"))

    def test_cancel_between_items_prevents_next_worker_start(self):
        self.app.queue.append(str(self.root / "next.wav"))
        started = []

        def cancel(_file, staging):
            started.append(_file)
            (Path(staging) / "lecture.txt").write_text("partial")
            self.app._cancel_processing()
            self.assertTrue(Path(staging).exists(), "UI must not clean while worker runs")
            return False

        self.app._run_worker = cancel
        self.app._process_queue()
        self.assertEqual(len(started), 1)
        self.assertEqual(self.statuses[-1], (0, "Cancelled"))
        self.assertEqual(list(self.root.iterdir()), [])
        self.assertEqual(self.finished, [True])

    def test_cancel_before_process_exists_still_sets_cancellation(self):
        self.app._cancel_processing()
        self.assertTrue(self.app.cancel_requested)

    def test_quiet_worker_cancellation_is_reaped_before_return(self):
        # This worker never logs, so cancellation cannot rely on a new stdout line.
        script = self.root / "quiet.py"
        script.write_text("import time\ntime.sleep(60)\n")
        self.app.queue_worker_args = [str(script), "--input", "", "--output-dir", ""]
        result = []
        thread = threading.Thread(target=lambda: result.append(
            self.app._run_worker("lecture.wav", str(self.root))))
        thread.start()
        deadline = time.monotonic() + 5
        while self.app.current_process is None and thread.is_alive() and time.monotonic() < deadline:
            time.sleep(0.01)
        process = self.app.current_process
        self.assertIsNotNone(process)
        self.app._cancel_processing()
        thread.join(timeout=5)
        if thread.is_alive():
            process.kill()
            thread.join(timeout=5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(result, [False])
        self.assertIsNotNone(process.poll())
        self.assertIsNone(self.app.current_process)

    def test_cancel_escalates_and_waits_if_worker_ignores_terminate(self):
        from unittest.mock import Mock
        process = Mock()
        process.poll.return_value = None
        process.wait.side_effect = [subprocess.TimeoutExpired("worker", 3), 0]
        with patch("app.os.killpg", create=True) as killpg:
            self.app._stop_worker(process)
        if os.name == "posix":
            self.assertGreaterEqual(killpg.call_count, 2)
        else:
            process.terminate.assert_called_once()
            process.kill.assert_called_once()
        self.assertEqual(process.wait.call_count, 2)

    @unittest.skipUnless(os.name == "posix", "macOS/Linux process-group behavior")
    def test_cancellation_stops_child_holding_stdout_open(self):
        script = self.root / "parent.py"
        script.write_text(
            "import subprocess, sys, time\n"
            "subprocess.Popen([sys.executable, '-c', "
            "'import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            "print(\"child-ready\", flush=True); time.sleep(60)'])\n"
            "time.sleep(60)\n"
        )
        self.app.queue_worker_args = [str(script), "--input", "", "--output-dir", ""]
        result = []
        thread = threading.Thread(target=lambda: result.append(
            self.app._run_worker("lecture.wav", str(self.root))))
        thread.start()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                if self.app.log_queue.get(timeout=0.1) == "child-ready":
                    break
            except queue.Empty:
                pass
        else:
            self.app.cancel_requested = True
            thread.join(timeout=5)
            self.fail("Test child did not start")
        self.app._cancel_processing()
        thread.join(timeout=5)
        self.assertFalse(thread.is_alive(), "Inherited stdout must not hang cancellation")
        self.assertEqual(result, [False])
        self.assertIsNone(self.app.current_process)


if __name__ == "__main__":
    unittest.main()
