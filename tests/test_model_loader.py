import unittest
from pathlib import Path

# Ensure project root is in sys.path
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.model_loader import _resolve_base_model_id, _load_adapter_config

class TestModelLoader(unittest.TestCase):
    def test_resolve_base_model_id(self):
        config = {"base_model_id": "google/gemma-3-4b-it"}
        model_id = _resolve_base_model_id(config)
        self.assertEqual(model_id, "google/gemma-3-4b-it")

    def test_load_adapter_config_nonexistent(self):
        config = _load_adapter_config(Path("non_existent_file.yaml"))
        self.assertEqual(config, {})

if __name__ == "__main__":
    unittest.main()
