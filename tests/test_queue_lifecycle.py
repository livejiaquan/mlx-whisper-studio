import unittest
from unittest.mock import Mock, patch

from app import ERROR, FileEntry, MLXWhisperApp


class QueueLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.app = MLXWhisperApp.__new__(MLXWhisperApp)
        self.app.root = Mock()
        self.app.is_running = False
        self.app.closing = False
        self.app.cancel_requested = False
        self.app.worker_thread = None
        self.app.current_process = None
        self.app.current_plan = None
        self.app.file_entries = [
            FileEntry("finished.wav", 1, "Done"),
            FileEntry("failed.wav", 1, "Failed"),
            FileEntry("cancelled.wav", 1, "Cancelled"),
        ]
        self.app.queue_entries = []
        self.app.queue_results = []
        self.app._post_ui = lambda callback: callback()
        self.app._selected_formats = lambda: ["txt"]
        self.app._build_worker_args = lambda *_: ["mlx_worker.py", "--input", "", "--output-dir", ""]
        self.app.output_dir_var = Mock()
        self.app.output_dir_var.get.return_value = ""
        self.app._set_status = Mock()
        self.app._log = Mock()
        for name in ("file_list", "start_button", "cancel_button", "retry_button",
                     "add_button", "remove_button", "clear_button", "progress", "current_file_label"):
            setattr(self.app, name, Mock())
        self.app.file_list.size.return_value = len(self.app.file_entries)

    def test_retry_starts_only_failed_and_cancelled_files(self):
        with patch("app.threading.Thread") as thread:
            self.app._start_queue(retry_only=True)
        self.assertEqual(self.app.queue, ["failed.wav", "cancelled.wav"])
        self.assertEqual(self.app.file_entries[0].status, "Done")
        self.assertEqual([entry.status for entry in self.app.queue_entries], ["Queued", "Queued"])
        self.assertTrue(self.app.is_running)
        thread.return_value.start.assert_called_once()
        for name in ("add_button", "remove_button", "clear_button", "retry_button"):
            getattr(self.app, name).configure.assert_called_with(state="disabled")

    def test_retry_index_updates_correct_visible_row(self):
        self.app.queue_entries = [self.app.file_entries[1]]
        self.app._mark_item_status(0, "Done")
        self.app.file_list.delete.assert_called_with(1)
        self.assertEqual(self.app.file_entries[0].status, "Done")
        self.assertEqual(self.app.file_entries[1].status, "Done")
        self.assertEqual(self.app.file_entries[2].status, "Cancelled")

    def test_running_queue_cannot_be_removed_cleared_or_appended(self):
        before = list(self.app.file_entries)
        self.app.is_running = True
        self.app._remove_selected()
        self.app._clear_files()
        self.app._add_files(["extra.wav"])
        self.assertEqual(self.app.file_entries, before)
        self.app.file_list.delete.assert_not_called()

    def test_failure_summary_is_not_reported_as_all_successful(self):
        self.app.queue_results = ["Done", "Failed", "Failed"]
        self.app.is_running = True
        self.app._finish_queue()
        self.app._set_status.assert_called_with(
            "Finished with errors: 1 succeeded, 2 failed, 0 cancelled", ERROR)
        self.assertFalse(self.app.is_running)
        self.app.retry_button.configure.assert_called_with(state="normal")

    def test_cancellation_summary_counts_unstarted_cancelled_jobs(self):
        self.app.queue_results = ["Done", "Cancelled", "Cancelled"]
        self.app.cancel_requested = True
        self.app._finish_queue()
        self.app._set_status.assert_called_with(
            "Cancelled: 1 succeeded, 0 failed, 2 cancelled", ERROR)

    def test_close_cancels_then_waits_for_worker_cleanup(self):
        self.app.worker_thread = Mock()
        self.app.worker_thread.is_alive.return_value = True
        self.app.is_running = True
        self.app._on_close()
        self.assertTrue(self.app.cancel_requested)
        self.assertTrue(self.app.closing)
        self.app.root.destroy.assert_not_called()
        self.app.root.after.assert_called_with(50, self.app._wait_for_close)
        self.app.worker_thread.is_alive.return_value = False
        self.app._wait_for_close()
        self.app.root.destroy.assert_called_once()

    def test_idle_window_closes_immediately(self):
        self.app._on_close()
        self.app.root.destroy.assert_called_once()

    def test_finish_while_closing_does_not_reenable_controls(self):
        self.app.closing = True
        self.app.cancel_requested = True
        self.app.queue_results = ["Cancelled"]
        self.app._finish_queue()
        for name in ("start_button", "retry_button", "add_button", "remove_button", "clear_button"):
            getattr(self.app, name).configure.assert_called_with(state="disabled")

    def test_repeated_start_does_not_launch_another_worker(self):
        self.app.is_running = True
        with patch("app.threading.Thread") as thread:
            self.app._start_queue()
        thread.assert_not_called()

    def test_thread_start_error_restores_controls_and_marks_jobs_retryable(self):
        with patch("app.threading.Thread") as thread:
            thread.return_value.start.side_effect = RuntimeError("no threads available")
            self.app._start_queue()
        self.assertFalse(self.app.is_running)
        self.assertEqual(self.app.queue_results, ["Failed"] * 3)
        self.app.start_button.configure.assert_called_with(state="normal")
        self.app.retry_button.configure.assert_called_with(state="normal")


if __name__ == "__main__":
    unittest.main()
