# Honesty rules (Ch.12 M3 device session):
# 1. Every number traces to a real subprocess: here, coremltools loading the
#    compiled .mlmodelc on this M3 and reading its MLComputePlan.
# 2. Only power is allowed to be estimated. Op placement is exact.
# 3. Unmeasured fields stay `null` with `# PLACEHOLDER`.
# 4. No new quantization concepts; the model came from Ch.11.
"""Report the fraction of compute ops scheduled on ANE / GPU / CPU.

This is the device analog of "verify the runtime actually executed where
you asked." Apple silicon routing is silent — there is no log line to
trust — so MLComputePlan is the only programmatic confirmation.

    python device/op_placement.py --mode run \\
        --model ~/models/efficientnet_lite0.mlpackage
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from _common import update_results, validate_file


_DEVICE_TO_UNIT = {                       #A coremltools 8.x device classes
    "MLNeuralEngineComputeDevice": "ane",
    "MLGPUComputeDevice":          "gpu",
    "MLCPUComputeDevice":          "cpu",
}


def _device_to_unit(dev_obj) -> str:
    name = type(dev_obj).__name__
    return _DEVICE_TO_UNIT.get(name, "cpu")


def _placement_fractions(model_path: Path) -> dict[str, float]:
    import coremltools as ct  #C lazy import keeps non-M3 environments importable
    # MLComputePlan needs the COMPILED .mlmodelc, not the .mlpackage source.
    mlmodel = ct.models.MLModel(str(model_path), compute_units=ct.ComputeUnit.ALL)
    compiled_path = mlmodel.get_compiled_model_path()
    plan = ct.models.compute_plan.MLComputePlan.load_from_path(
        compiled_path, compute_units=ct.ComputeUnit.ALL
    )

    program = plan.model_structure.program
    if program is None:
        raise SystemExit("model has no mlProgram section — cannot inspect placement")

    main_fn = program.functions.get("main") if hasattr(program, "functions") else program["main"]

    counts: Counter[str] = Counter()
    for op in main_fn.block.operations:
        usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
        if usage is None:
            counts["cpu"] += 1                            #B treat unscheduled as cpu fallback
            continue
        pref = getattr(usage, "preferred_compute_device", None)
        counts[_device_to_unit(pref) if pref is not None else "cpu"] += 1

    total = sum(counts.values())
    if total == 0:
        raise SystemExit("compute plan reported zero ops; refusing to write zeros")
    return {u: round(counts.get(u, 0) / total, 4) for u in ("ane", "gpu", "cpu")}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="MLComputePlan op-placement fractions")
    ap.add_argument("--mode", choices=["run"], default="run")
    ap.add_argument("--model", required=True)
    args = ap.parse_args(argv)

    model_path = validate_file(args.model, "model")
    frac = _placement_fractions(model_path)
    update_results({"device": {"op_placement_frac": frac}})
    print(f"[op_placement] ane={frac['ane']:.3f}  gpu={frac['gpu']:.3f}  cpu={frac['cpu']:.3f}")


if __name__ == "__main__":
    main()
