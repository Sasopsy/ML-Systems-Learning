"""Compare saved CUDA dX results against a CPU float64 reference."""

import json
from pathlib import Path

import torch


def main():
    directory = Path(__file__).resolve().parent / "results" / "conv_dgrad" / "tensors"
    filenames = {
        "off": "conv_dgrad_off_no_tf32_007.pt",
        "on": "conv_dgrad_on_no_tf32_008.pt",
    }
    saved = {
        mode: torch.load(directory / name, map_location="cpu", weights_only=True)
        for mode, name in filenames.items()
    }
    for key in ("weight", "grad_output"):
        assert torch.equal(saved["off"][key], saved["on"][key]), key
    for tensors in saved.values():
        for tensor in tensors.values():
            assert tensor.dtype == torch.float32
            assert torch.isfinite(tensor).all()
        assert tuple(tensors["grad_input"].shape) == (32, 512, 2, 2)

    # Promote the original input values; do not generate new float64 inputs.
    baseline = saved["off"]
    reference = torch.nn.grad.conv2d_input(
        (32, 512, 2, 2),
        baseline["weight"].double(),
        baseline["grad_output"].double(),
        stride=1,
        padding=1,
        dilation=1,
        groups=1,
    )
    assert reference.dtype == torch.float64
    assert torch.isfinite(reference).all()

    # Preserve the original float32 acceptance criterion after promotion.
    atol, rtol = 1e-5, 1.3e-6
    allowed = atol + rtol * reference.abs()
    report = {
        "reference": "CPU float64 conv2d_input; not exact arithmetic",
        "inputs_exactly_equal": True,
        "atol": atol,
        "rtol": rtol,
        "comparisons": {},
    }

    # Editable internal boundaries; include both tails automatically.
    buckets = [0.1, 1, 10]
    if any(not (0 < boundary < float("inf")) for boundary in buckets):
        raise ValueError("Bucket boundaries must be positive and finite.")
    if any(left >= right for left, right in zip(buckets, buckets[1:])):
        raise ValueError("Bucket boundaries must be strictly increasing.")
    edges = [0.0, *buckets, float("inf")]

    for mode, tensors in saved.items():

        error = (tensors["grad_input"].double() - reference).abs()
        failed = error > allowed
        mismatches = int(failed.sum().item())
        relative_l2_error = (
            torch.linalg.norm(tensors["grad_input"].double() - reference)
            / torch.linalg.norm(reference)
        ).item()
        magnitude = reference.abs()
        bin_results = {}
        for lower, upper in zip(edges, edges[1:]):
            mask = (lower <= magnitude) & (magnitude < upper)
            total_count = mask.sum().item()
            failed_count = (failed & mask).sum().item()
            bin_results[f"[{lower:g}, {upper:g})"] = {
                "total_count": total_count,
                "failed_count": failed_count,
                "failure_rate_percent": (
                    100 * failed_count / total_count if total_count else None
                ),
            }
        assert sum(group["total_count"] for group in bin_results.values()) == error.numel()
        assert sum(group["failed_count"] for group in bin_results.values()) == mismatches

        report["comparisons"][mode] = {
            "source": filenames[mode],
            "max_absolute_error": error.max().item(),
            "mean_absolute_error": error.mean().item(),
            "mismatched_elements": mismatches,
            "total_elements": error.numel(),
            "passes_original_tolerance": mismatches == 0,
            "relative_l2_error": relative_l2_error,
            "bins": bin_results,
        }
    # Report both modes even if numerical acceptance fails for either one.
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
