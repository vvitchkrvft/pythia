from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pythia.registry import ModelRegistry


class ModelRegistryTests(unittest.TestCase):
    def test_get_matches_alias_first_then_model_id(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.yaml"
            config_path.write_text(
                "models:\n"
                "  - name: alpha\n"
                "    model_id: repo/alpha\n"
                "  - name: repo/alpha\n"
                "    model_id: repo/beta\n",
                encoding="utf-8",
            )
            registry = ModelRegistry(config_path)

            self.assertEqual(registry.get("alpha").model_id, "repo/alpha")
            self.assertEqual(registry.get("repo/alpha").name, "repo/alpha")
            self.assertEqual(registry.get("repo/beta").name, "repo/alpha")


if __name__ == "__main__":
    unittest.main()
