"""Bake the sorting demo's vocabulary into the two vision models. Runs HERE, on the laptop.

The CoreIO service is two stages (see `service/vision_server.py`):

1. **segment** — `yoloe-11s-seg-pf`, YOLOE's prompt-free model. It separates every object
   in the frame into its own mask whatever it is, with no vocabulary to maintain.
2. **classify** — `yoloe-11s-seg` with *this* demo's prompts baked in. It is what decides
   teddy bear, bottle or other, and which grip material each object gets.

The classify stage takes its classes as text, which needs a CLIP-style text encoder
(MobileCLIP) to turn prompts into embeddings. The CoreIO has no route to download that
encoder, so the text side runs once here: `set_classes()` computes the embeddings and
`save()` writes a plain `.pt` that already carries them.

**Run it with ultralytics 8.3.156** — the version inside the robot's `yolo:latest` image.
A checkpoint pickled by a newer ultralytics does not reliably unpickle on an older one::

    uv venv --python 3.12 runtime/bake-venv
    uv pip install --python runtime/bake-venv/bin/python ultralytics==8.3.156 \\
        torch torchvision --index-url https://download.pytorch.org/whl/cpu \\
        --extra-index-url https://pypi.org/simple
    runtime/bake-venv/bin/python coreio/vision/bake_vocab.py

Writes `runtime/weights/vision/{segment.pt, classify.pt, vocab.json}`, which is exactly
what `coreio/deploy_vision.py` ships.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
_MATERIALS = ("soft", "medium", "hard")


def load_vocab(path: Path) -> dict:
    """Flatten a vocabulary file into prompts, label -> group and label -> material.

    Order is preserved and matters: the baked embedding matrix is indexed by position, so
    the Nth prompt here is class N on the robot.
    """
    spec = json.loads(path.read_text())
    groups = spec.get("groups") or {}
    if not groups:
        sys.exit(f"{path}: no 'groups' — nothing to bake")

    prompts: list[str] = []
    label_to_group: dict[str, str] = {}
    for group, labels in groups.items():
        for label in (item.strip() for item in labels):
            if not label:
                continue
            if label in label_to_group:
                sys.exit(f"{path}: '{label}' is in both '{label_to_group[label]}' and '{group}'")
            label_to_group[label] = group
            prompts.append(label)

    defaults = spec.get("material_defaults", {})
    label_to_material = {label: defaults.get(group, "medium") for label, group in label_to_group.items()}
    for material, labels in (spec.get("materials") or {}).items():
        if material not in _MATERIALS:
            sys.exit(f"{path}: material '{material}' is not one of {_MATERIALS}")
        for label in labels:
            if label not in label_to_group:
                sys.exit(f"{path}: material list names '{label}', which is not a prompt")
            label_to_material[label] = material

    ignore = list(spec.get("ignore_groups", []))
    unknown = [g for g in ignore if g not in groups]
    if unknown:
        sys.exit(f"{path}: ignore_groups names {unknown}, which are not groups")
    return {
        "vocab": path.name,
        "fallback_group": spec.get("fallback_group", "other"),
        "ignore_groups": ignore,
        "material_defaults": defaults,
        "prompts": prompts,
        "label_to_group": label_to_group,
        "label_to_material": label_to_material,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vocab", type=Path, default=_HERE / "vocab" / "sort3.json")
    parser.add_argument("--segment-model", default="yoloe-11s-seg-pf.pt",
                        help="prompt-free segmenter (stage 1)")
    parser.add_argument("--classify-model", default="yoloe-11s-seg.pt",
                        help="text-promptable model the vocabulary is baked into (stage 2)")
    parser.add_argument("--out-dir", type=Path, default=_ROOT / "runtime" / "weights" / "vision")
    args = parser.parse_args()

    import ultralytics

    if ultralytics.__version__ != "8.3.156":
        print(f"[bake] WARNING: ultralytics {ultralytics.__version__} here, 8.3.156 on the robot — "
              f"the baked checkpoint may not load there. See this file's docstring.")

    spec = load_vocab(args.vocab)
    counts: dict[str, int] = {}
    for group in spec["label_to_group"].values():
        counts[group] = counts.get(group, 0) + 1
    print(f"[bake] {args.vocab.name}: {len(spec['prompts'])} prompts across {counts}")

    from ultralytics import YOLOE

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache = args.out_dir / "base"
    cache.mkdir(exist_ok=True)

    # Stage 1 needs no baking — it is used as shipped — but it is loaded once here so the
    # download happens on the laptop, where there is an internet connection.
    segment_src = cache / args.segment_model
    if not segment_src.is_file():
        YOLOE(args.segment_model)
        shutil.move(args.segment_model, segment_src)
    shutil.copy(segment_src, args.out_dir / "segment.pt")
    print(f"[bake] segment stage: {args.segment_model}")

    classify_src = cache / args.classify_model
    if not classify_src.is_file():
        YOLOE(args.classify_model)
        shutil.move(args.classify_model, classify_src)
    print(f"[bake] classify stage: encoding {len(spec['prompts'])} prompts into {args.classify_model} (CPU)...")
    model = YOLOE(str(classify_src))
    model.set_classes(spec["prompts"], model.get_text_pe(spec["prompts"]))
    # `get_text_pe` caches the text encoder on the model. Pickled into the checkpoint it
    # would drag MobileCLIP onto the robot — the very dependency this script exists to
    # keep off it. The embeddings are already in the head, so dropping it loses nothing.
    if hasattr(model.model, "clip_model"):
        model.model.clip_model = None
    model.save(str(args.out_dir / "classify.pt"))

    spec["segment_model"] = args.segment_model
    spec["classify_model"] = args.classify_model
    (args.out_dir / "vocab.json").write_text(json.dumps(spec, indent=2))
    print(f"[bake] wrote {args.out_dir}/segment.pt, classify.pt, vocab.json")
    print("[bake] next: python coreio/deploy_vision.py")


if __name__ == "__main__":
    main()
