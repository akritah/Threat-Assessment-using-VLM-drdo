import unittest
import os
import tempfile
from pathlib import Path
import yaml

# Ensure project root is in sys.path
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.config import TrainingConfig

class TestConfiguration(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_default_config(self):
        config = TrainingConfig()
        self.assertEqual(config.base_model_id, "google/gemma-3-4b-it")
        self.assertEqual(config.lora_r, 16)
        self.assertTrue(config.load_in_4bit)

    def test_from_yaml(self):
        yaml_content = {
            "base_model_id": "google/gemma-3-4b-it",
            "output_dir": "adapters/test_run",
            "num_train_epochs": 5,
            "learning_rate": 5.0e-5,
            "lora_target_modules": ["q_proj", "v_proj"]
        }
        yaml_file = self.temp_path / "test_config.yaml"
        with yaml_file.open("w", encoding="utf-8") as f:
            yaml.safe_dump(yaml_content, f)

        config = TrainingConfig.from_yaml(yaml_file)
        self.assertEqual(config.base_model_id, "google/gemma-3-4b-it")
        self.assertEqual(config.output_dir, "adapters/test_run")
        self.assertEqual(config.num_train_epochs, 5)
        self.assertEqual(config.learning_rate, 5.0e-5)
        self.assertEqual(config.lora_target_modules, ["q_proj", "v_proj"])

    def test_path_resolution(self):
        config = TrainingConfig(output_dir="adapters/custom")
        expected_path = PROJECT_ROOT / "adapters" / "custom"
        self.assertEqual(config.output_path, expected_path)

if __name__ == "__main__":
    unittest.main()
