"""Regenerate goldens/schema.json from the pydantic model (used for VS Code validation)."""

import json
from pathlib import Path

from mfg_rag_eval.models import GoldenDataset

out = Path(__file__).resolve().parent.parent / "goldens" / "schema.json"
out.write_text(json.dumps(GoldenDataset.model_json_schema(), indent=2) + "\n", encoding="utf-8")
print(f"wrote {out}")
