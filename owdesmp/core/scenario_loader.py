"""Minimal scenario loader for slice 1.

Parses the JSON file and returns it as a dict. No validation beyond what the
simulator needs at this stage. Later slices add a schema check (jsonschema or
pydantic) and resolve mobility/scanner/channel sections.
"""

import json
from pathlib import Path


def load_scenario(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
