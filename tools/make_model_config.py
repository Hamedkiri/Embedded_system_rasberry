#!/usr/bin/env python3
import json, sys
from pathlib import Path

"""
Usage:
  python tools/make_model_config.py \
     --hparams configs/mtm_hparams.json \
     --labels classes/weather_labels.json \
     --type pytorch \
     --input_size 224 \
     --out_dir models
"""

def parse_args():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--hparams", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--type", default="pytorch", choices=["pytorch","onnx","tflite"])
    ap.add_argument("--input_size", type=int, default=224)
    ap.add_argument("--use_attention", action="store_true")
    ap.add_argument("--attn_token_dim", type=int, default=None)
    ap.add_argument("--cls_hidden_dims", type=int, nargs="*", default=[])
    ap.add_argument("--cls_num_layers", type=int, default=0)
    ap.add_argument("--out_dir", default="models")
    return ap.parse_args()

def main():
    args = parse_args()
    hp = json.loads(Path(args.hparams).read_text(encoding="utf-8"))
    labels = json.loads(Path(args.labels).read_text(encoding="utf-8"))

    # Validation: tailles
    mismatches = []
    for task, n in hp.get("tasks", {}).items():
        if task not in labels:
            mismatches.append(f"'{task}' absent des labels")
        else:
            if int(n) != len(labels[task]):
                mismatches.append(f"'{task}': attendu {n} classes, labels={len(labels[task])}")
    if mismatches:
        print("[WARN] Mismatches:\n - " + "\n - ".join(mismatches))

    cfg = {
        "type": args.type,
        "input_size": int(args.input_size),
        "tasks": labels,
        "extra": {
            "truncate_after_layer": int(hp.get("truncate_layer", 10)),
            "use_attention": bool(args.use_attention or True),  # par défaut True
            "attn_token_dim": args.attn_token_dim,
            "cls_hidden_dims": args.cls_hidden_dims,
            "cls_num_layers": int(args.cls_num_layers)
        }
    }

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(f"✓ Écrit: {out_dir/'config.json'}")

if __name__ == "__main__":
    main()
