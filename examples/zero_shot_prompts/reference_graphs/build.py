"""Stamp pick-single reference graphs from the libero_quickstart template.

For each ``pick`` case in the registry, copy
``examples/libero_quickstart/graph`` to ``_generated/<case>/`` and patch the
two perceive nodes' ``object_name`` (target + container) to the case's
descriptions. The checkpoints/ dir is dropped (it hardcodes the soup can and
is advisory-only under ``checkpoints=warn`` — our criterion is the judge).

    uv run python examples/zero_shot_prompts/reference_graphs/build.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_EXAMPLE = _HERE.parent
_REPO = _EXAMPLE.parents[1]
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from descriptions import describe  # noqa: E402
from registry import pick_refs  # noqa: E402

TEMPLATE = _REPO / "examples" / "libero_quickstart" / "graph"
GEN_DIR = _HERE / "_generated"


def _patch_perceive_names(wf: dict, target_desc: str, container_desc: str) -> None:
    for sg_name, sg in wf.get("subgraphs", {}).items():
        for node in sg.get("nodes", {}).values():
            if node.get("type") != "script":
                continue
            inp = node.get("inputs", {})
            if "object_name" not in inp:
                continue
            if sg_name.startswith("target"):
                inp["object_name"] = target_desc
            elif sg_name.startswith("container"):
                inp["object_name"] = container_desc


def build() -> list[str]:
    GEN_DIR.mkdir(parents=True, exist_ok=True)
    built = []
    for ref in pick_refs():
        dest = GEN_DIR / ref.case
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(TEMPLATE, dest,
                        ignore=shutil.ignore_patterns("__pycache__",
                                                      "checkpoints", "*.pyc"))
        wf = json.loads((dest / "workflow.json").read_text())
        wf["meta"]["name"] = f"pick_{ref.target}"
        wf["meta"]["description"] = (
            f"Reference: pick the {describe(ref.target)} and place it in the "
            f"{describe(ref.container)}."
        )
        _patch_perceive_names(wf, describe(ref.target), describe(ref.container))
        (dest / "workflow.json").write_text(json.dumps(wf, indent=1) + "\n")
        built.append(ref.case)
    return built


if __name__ == "__main__":
    names = build()
    print(f"stamped {len(names)} pick-single reference graphs into "
          f"{GEN_DIR.relative_to(_REPO)}:")
    for n in names:
        print(f"  {n}")
