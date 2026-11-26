"""Model sizing and the ED screening protocol, without numerical imports."""

import math
from statistics import median


PHASES = {"sf": 2.0, "bkt": 3.275, "mott": 8.0}
LEGACY_MODELS = (
    "FCNet", "RBMNet", "Conv1DNet", "Conv1DPoolNet", "TransformerNet",
    "ConvTransformer", "ViTNet",
)
SCREEN_MODELS = (*LEGACY_MODELS, "InvariantConv1D")
MODELS = (*SCREEN_MODELS, "JastrowNet")
ATTENTION = ("TransformerNet", "ConvTransformer", "ViTNet")
BANDS = (2_000, 8_000, 32_000)
PARAMETER_CAP = BANDS[-1]
LEARNING_RATES = (3e-4, 1e-3, 3e-3)
TUNING_SEED = 123
CONFIRM_SEEDS = (101, 202, 303)
GATE = 1e-3
TIGHT_TARGET = 1e-4
NEAR_MISS = 2e-3
SCREEN_STEPS = 3_000
SCREEN_FLOPS = {8: 50_000_000_000_000, 10: 200_000_000_000_000}
CHECKPOINT_FRACTIONS = (0.1, 0.25, 0.5, 0.75, 1.0)


def model_cost(arch, length, nmax, width):
    """Return exact parameter count and a forward FLOP estimate per configuration.

    Count multiply-adds as two and include attention score/value products.
    Pointwise activations, normalization and data movement are outside the estimate.
    """
    channels = nmax + 1
    visible = length * channels
    if arch == "JastrowNet":
        return channels + (length // 2 + 1) * channels**2, 2 * visible + length * (length + 1) // 2
    if arch == "RBMNet":
        return visible + (visible + 1) * width, 2 * (visible + visible * width)
    if arch == "FCNet":
        return (visible + 1) * width + (width + 1) * width + width + 1, 2 * (
            visible * width + width**2 + width
        )

    conv_params = (3 * channels + 1) * width
    conv_flops = 6 * length * channels * width
    output_params = width + 1
    if arch == "Conv1DPoolNet":
        return conv_params + 2 * (3 * width + 1) * width + (width + 1) * width + output_params, (
            conv_flops + 6 * (length // 2 + length // 4) * width**2 + 2 * (width**2 + width)
        )
    if arch == "InvariantConv1D":
        return conv_params + (3 * width + 1) * width + output_params, (
            conv_flops + 6 * length * width**2 + length * width + 2 * width
        )

    flat_params = (length * width + 1) * width + output_params
    flat_flops = 2 * (length * width**2 + width)
    if arch == "Conv1DNet":
        return conv_params + flat_params, conv_flops + flat_flops

    layer_params = 8 * width**2 + 11 * width
    if arch == "ViTNet":
        tokens = length + 1
        params = (channels + 1) * width + (length + 2) * width + 2 * layer_params + 2 * width + output_params
        flops = 2 * length * channels * width + 2 * attention_flops(tokens, width) + 2 * width
        return params, flops
    if arch == "ConvTransformer":
        return conv_params + layer_params + flat_params, conv_flops + attention_flops(length, width) + flat_flops
    return (channels + 1) * width + layer_params + flat_params, (
        2 * length * channels * width + attention_flops(length, width) + flat_flops
    )


def attention_flops(tokens, width):
    return 16 * tokens * width**2 + 4 * tokens**2 * width


def width_for_band(arch, length, nmax, band):
    increment = 4 if arch in ATTENTION else 1
    options = []
    for width in range(increment, PARAMETER_CAP + 1, increment):
        params, _ = model_cost(arch, length, nmax, width)
        if params > PARAMETER_CAP:
            break
        options.append((abs(params - band), width))
        if params >= band:
            break
    return min(options)[1] if options else None


def sector_dimension(length, nmax):
    return sum(
        (-1)**k * math.comb(length, k) * math.comb(2 * length - 1 - k * (nmax + 1), length - 1)
        for k in range(length // (nmax + 1) + 1)
    )


def cost_estimate(config):
    params, forward = model_cost(config["arch"], config["length"], config["nmax"], config["width"])
    dimension = sector_dimension(config["length"], config["nmax"])
    # One no-grad pass, one gradient replay forward and two forward-equivalents
    # for its backward. Include a fixed arithmetic allowance for reductions and Adam.
    return {
        "parameters": params,
        "dimension": dimension,
        "forward_flops_per_config": forward,
        "estimated_flops_per_step": dimension * (4 * forward + 24) + 16 * params,
        "estimated_flops_per_score": dimension * (forward + 16),
    }


def tuning_trials(architectures, options, output_root):
    trials, unavailable = [], []
    length, nmax = options["length"], options["nmax"]
    for arch in architectures:
        prior_widths = set()
        for band in BANDS:
            width = width_for_band(arch, length, nmax, band)
            if width is None or width in prior_widths:
                unavailable.append({"arch": arch, "band": band, "reason": "no distinct width within cap"})
                continue
            prior_widths.add(width)
            for lr in LEARNING_RATES:
                base = dict(options, arch=arch, width=width, band=band, lr=lr, seed=TUNING_SEED,
                            max_steps=SCREEN_STEPS // 4, max_flops=SCREEN_FLOPS[length] // 4)
                trials.extend(phase_trials(base, output_root / arch / f"band-{band}" / f"lr-{lr:g}"))
    return trials, unavailable


def phase_trials(base, output_root):
    return [dict(base, phase=phase, output=str(output_root / phase)) for phase in PHASES]


def tuning_choices(records):
    groups = {}
    for config, result in records:
        key = (config["arch"], config["band"], config["lr"])
        groups.setdefault(key, []).append((config, result))
    ranked = {}
    for (arch, band, lr), trials in groups.items():
        if {config["phase"] for config, _ in trials} != set(PHASES) or len(trials) != len(PHASES):
            raise ValueError(f"incomplete or duplicate tuning phases for {arch}, band {band}, lr {lr}")
        config = trials[0][0]
        worst = max(result["best"]["infidelity"] for _, result in trials)
        choice = (worst, cost_estimate(config)["parameters"], lr)
        if arch not in ranked or choice < ranked[arch][0]:
            ranked[arch] = (choice, config)
    return [(arch, score[0], config) for arch, (score, config) in ranked.items()]


def confirmation_trials(choices, output_root):
    trials = []
    for arch, _, config in choices:
        for seed in CONFIRM_SEEDS:
            base = dict(config, seed=seed, max_steps=SCREEN_STEPS, max_flops=SCREEN_FLOPS[config["length"]])
            trials.extend(phase_trials(base, output_root / arch / f"seed-{seed}"))
    return trials


def confirmation_report(records, expected_trials):
    lengths = {config["length"] for config in expected_trials}
    if len(lengths) != 1:
        raise ValueError("a screening manifest must contain a single ring length")
    length = lengths.pop()
    grouped = {}
    for config, result in records:
        grouped.setdefault((config["arch"], config["phase"]), {})[config["seed"]] = result
    report = {}
    for arch in dict.fromkeys(config["arch"] for config in expected_trials):
        phases = {}
        for phase in PHASES:
            results = grouped.get((arch, phase), {})
            seed_scores = {seed: result["best"]["infidelity"] for seed, result in results.items()}
            complete = set(seed_scores) == set(CONFIRM_SEEDS)
            middle = median(seed_scores.values()) if complete else None
            phases[phase] = {
                "median_infidelity": middle,
                "default_pass": middle <= GATE if complete else None,
                "tight_pass": middle <= TIGHT_TARGET if complete else None,
                "seed_infidelities": seed_scores,
                "missing_seeds": sorted(set(CONFIRM_SEEDS) - seed_scores.keys()),
            }
        passes = [phase["default_pass"] for phase in phases.values()]
        complete = None not in passes
        report[arch] = {
            "phases": phases,
            f"l{length}_all_three_pass": all(passes) if complete else None,
            "bounded_retest_eligible": (
                not all(passes) and all(phase["median_infidelity"] <= NEAR_MISS for phase in phases.values())
            ) if complete else None,
        }
    return report
