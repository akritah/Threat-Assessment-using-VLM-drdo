import unittest
import tempfile
from pathlib import Path
import cv2
import numpy as np

# Ensure project root is in sys.path
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.frame_extractor import extract_frames

class TestFrameExtractor(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        
        # Write a tiny 20-frame dummy video
        self.video_path = self.temp_path / "test_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(self.video_path), fourcc, 10.0, (100, 100))
        for i in range(20):
            # Draw simple pattern on frame
            frame = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.circle(frame, (50, 50), i, (255, 255, 255), -1)
            out.write(frame)
        out.release()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            extract_frames(self.temp_path / "non_existent.mp4", self.temp_path / "out", 6)

    def test_unsupported_extension(self):
        invalid_file = self.temp_path / "test.txt"
        invalid_file.touch()
        with self.assertRaises(ValueError):
            extract_frames(invalid_file, self.temp_path / "out", 6)

    def test_invalid_frame_count(self):
        with self.assertRaises(ValueError):
            extract_frames(self.video_path, self.temp_path / "out", 0)

    def test_successful_extraction(self):
        out_dir = self.temp_path / "extracted"
        frame_paths = extract_frames(self.video_path, out_dir, 6)
        
        # Verify 6 frames were returned and exist
        self.assertEqual(len(frame_paths), 6)
        for path in frame_paths:
            self.assertTrue(path.exists())
            self.assertEqual(path.suffix, ".jpg")

if __name__ == "__main__":
    unittest.main()
