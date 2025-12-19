from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd: list[str]) -> None:
    print("\n>>>", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=10, help="number of repeats (default: 10)")
    p.add_argument("--base-out", type=str, default="results/qi_fedavg", help="base output dir")
    #p.add_argument("--strategy", type=str, default="qi", help="server strategy name")
    #p.add_argument("--non-iid", action="store_true", help="enable non-iid partitioning")
    #p.add_argument("--fraction-train", type=float, default=1.0)
    #p.add_argument("--attacker-enabled", action="store_true", help="keep OFF for normal FedAvg")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    base_out = Path(args.base_out)
    base_out.mkdir(parents=True, exist_ok=True)

    # Save a top-level manifest for reproducibility
    manifest = {
        "seeds": args.seeds,
      #  "strategy": args.strategy,
      #  "non_iid": args.non_iid,
       # "fraction_train": args.fraction_train,
       # "attacker_enabled": args.attacker_enabled,
        "cwd": str(Path.cwd()),
    }
    (base_out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    for seed in range(args.seeds):
        run_dir = base_out / f"seed_{seed:02d}"
        qi_dir = run_dir / "qi"
        #gtg_dir = run_dir / "gtg"
        done_flag = run_dir / "DONE.json"

        # Skip if already finished (so you can resume safely)
        if done_flag.exists():
            print(f"[skip] seed {seed} already done: {done_flag}")
            continue

        qi_dir.mkdir(parents=True, exist_ok=True)
       # gtg_dir.mkdir(parents=True, exist_ok=True)

        # IMPORTANT: each run must write to unique folders
        # Also set partition-seed so you can reproduce splits per run
        run_config = {
           # "strategy": args.strategy,
           # "fraction-train": args.fraction_train,
           # "attacker-enabled": args.attacker_enabled,
           # "non-iid": args.non_iid,
            "partition-seed": seed,
            #"qi-out-dir": str(qi_dir).replace("\\", "/"),
            #"gtg-out-dir": str(gtg_dir).replace("\\", "/"),
            "qi-out-dir": str(qi_dir),              # string/path -> quoted
            #"gtg-out-dir": str(gtg_dir),            # string/path -> quoted
        }

       # if args.strategy in ["gtg", "qi_gtg", "qi+gtg", "compare"]:
        #    run_config["gtg-out-dir"] = str(gtg_dir)

        # 🔒 SAVE CONFIG FOR THIS SEED (HERE)
        (run_dir / "run_config.json").write_text(
            json.dumps(run_config, indent=2),
            encoding="utf-8",
        )

        # Flower CLI expects a single string "k=v k=v ..."
        #run_config_str = " ".join([f"{k}={v}" for k, v in run_config.items()])

        def toml_value(val):
            # booleans
            if isinstance(val, bool):
                return "true" if val else "false"
            # numbers
            if isinstance(val, (int, float)):
                return str(val)
            # strings / paths -> TOML string (quoted)
            return json.dumps(str(val).replace("\\", "/"))

        #run_config_str = " ".join([f"{k}={toml_value(v)}" for k, v in run_config.items()])
        run_config_str = " ".join(
            f"{k}={toml_value(val)}" for k, val in run_config.items()
        )

        flwr_cmd = ["flwr", "run", ".", "--run-config", run_config_str]

        if args.dry_run:
            print("[dry-run] would run:", " ".join(flwr_cmd))
            continue

        # 1) Run federated training (this should create qi logs + gtg logs)
        #run_cmd(flwr_cmd)

        try:
            run_cmd(flwr_cmd)
        except subprocess.CalledProcessError:
            print(f"[fail] FL run crashed for seed={seed}, skipping offline QI.")
            continue

        # 2) Run offline QI scoring and store it inside the same seed folder
        # Adjust this import path if your module name differs.
        offline_qi_cmd = [
            sys.executable,
            "-c",
            (
                "from pytorchexample.offline_qi import run_offline_qi; "
                f"run_offline_qi(qi_dir=r'{str(qi_dir)}', "
                f"out_path=r'{str(qi_dir / 'offline_qi_scores.json')}')"
            ),
        ]
        run_cmd(offline_qi_cmd)

        # Mark done (so you can resume later without rerunning seeds)
        done_flag.write_text(json.dumps({"seed": seed, "status": "ok"}, indent=2), encoding="utf-8")
        print(f"[ok] seed {seed} finished -> {run_dir}")


if __name__ == "__main__":
    main()
