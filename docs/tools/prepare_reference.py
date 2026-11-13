"""Render the public benchmark page from retained JSON, without runtime imports."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DESTINATION = ROOT / 'docs/site/benchmarks.md'
REPOSITORY = 'https://github.com/angkun-research/BoseHubbardNQS/blob/main/'


def render():
    one_d = json.loads((ROOT / 'docs/benchmarks/2026-10-30/results.json').read_text())
    group = json.loads((ROOT / 'pythonTest/2D/docs/gcnn-study-results.json').read_text())
    lines = [
        '# Benchmarks and limitations', '',
        'These are retained finite-budget experiments. They establish runnable',
        'workflows and measured finite-system accuracy, not convergence or a',
        'universal ranking of network architectures. The tables are generated from',
        'the recorded JSON files; building this page performs no simulation.', '',
        '## 1D: eight-site exact-sector evaluation', '',
        'Periodic rings, unit filling, cutoff 4, t=1, U=2 in the 1D convention',
        '`U*n*(n-1)`, mu=0. All four models used 100 native-gradient/Adam updates,',
        'width 32 where applicable, and training seed 123. The exact sector has',
        '5,475 configurations; its ED energy is -7.4597647030t.', '',
        '| Ansatz | Parameters | Exact model energy/site (t) | Absolute overlap |',
        '| --- | ---: | ---: | ---: |',
    ]
    for row in one_d['runs']:
        if row['L'] == 8:
            lines.append(f"| {row['arch']} | {row['parameters']} | {row['energy_per_site']:.9f} | {row['overlap']:.9f} |")
    lines += [
        '', 'The overlap column is unsquared. All four states remain above ED.',
        'These short runs do not establish each architecture’s best achievable accuracy.', '',
        '## 1D: sixteen-site frozen sampling', '',
        'The same 100-update protocol was evaluated using two fresh chains per',
        'checkpoint, seeds 456 and 789, with 128 walkers, 4,096 burn-in proposals',
        'per walker and 256 retained steps. No exact reference was computed.', '',
        '| Ansatz | Energy/site, seed456 (t) | Energy/site, seed789 (t) |',
        '| --- | ---: | ---: |',
    ]
    for row in one_d['runs']:
        if row['L'] == 16:
            first, second = row['repeats']
            lines.append(f"| {row['arch']} | {first['energy']/16:.9f} | {second['energy']/16:.9f} |")
    lines += [
        '', 'These are repeated measurements of the same trained weights, not',
        'independent training seeds. Sampling variation and unresolved mixing',
        'prevent a precise architecture ranking.', '',
        f'[Full 1D protocol and interpretation]({REPOSITORY}docs/VMC_BENCHMARKS.md).', '',
        '## 2D: the same minimal GCNN across sizes and objectives', '',
        'The width-8 GCNN has 4,993 parameters: one lifting convolution, one group',
        'convolution, SiLU and a mean/linear scalar readout. All cases use unit',
        'filling, cutoff 4, t=1, U=16.74 in the 2D `U*n*(n-1)/2` convention,',
        'mu=0 and periodic boundaries. Squares have D4 symmetry; rectangles',
        'retain D4-tied filters and D2 lattice symmetry.', '',
        'Supervised fitting used 10,000 Adam/log-MSE updates. VMC used 500',
        'centered ridge-SR/SGD updates. Each case has one training seed.', '',
        '| Case | Absolute overlap | Squared fidelity | Exact energy error/site (t) |',
        '| --- | ---: | ---: | ---: |',
    ]
    supervised = {row['shape']: row for row in group['supervised']}
    for shape in ('2x3', '3x3', '3x4'):
        row = supervised[shape]
        lines.append(f"| {shape} log-MSE | {row['amplitude_overlap_abs_unsquared']:.10f} | {row['fidelity_squared']:.10f} | {row['energy_error_per_site']:.8g} |")
    exact = next(row['exact_saved_state'] for row in group['vmc'] if row['shape'] == '3x3')
    lines.append(f"| 3x3 VMC | {exact['amplitude_overlap_abs_unsquared']:.10f} | {exact['fidelity_squared']:.10f} | {exact['energy_error_per_site']:.8g} |")
    lines += [
        '', '**The 3x4 fit misses the previous 0.99 absolute-overlap working',
        'criterion.** Its loss was still improving at the fixed update budget;',
        'the miss does not prove a capacity limit. The rectangular saved-checkpoint',
        'fidelities were independently recomputed over the full ED sectors.', '',
        '## 2D: larger-system frozen energies', '',
        'Fresh measurements use seeds 321 and 654, 64 walkers, 4,096 burn-in',
        'proposals per walker and 256 retained steps, with thinning equal to the',
        'number of sites. The records retain energy reblocking at lengths',
        '16, 32, 64 and 128.', '',
        '| Lattice | Energy/site, seed321 (t) | Energy/site, seed654 (t) |',
        '| --- | ---: | ---: |',
    ]
    for row in group['vmc']:
        first, second = row['frozen']
        lines.append(f"| {row['shape']} | {first['energy_per_site']:.9f} | {second['energy_per_site']:.9f} |")
    lines += [
        '', 'No exact fidelity reference is available for 4x4 or 5x5. Remaining',
        'local-energy variance and finite training budgets prevent a claim of',
        'converged ground-state accuracy. Block errors are conditional estimates,',
        'not confidence intervals or an effective-sample-size measurement.', '',
        'The initial aggressive 5x5 pilot failed with nonfinite local energies.',
        'The retained results use the conservative protocol recorded in the full',
        'report. Exact 3x5 and 4x5 fits were replaced with 2x3 because of the',
        'existing full-ED resource limits; a new orbit-reduced solver was deferred.', '',
        '## Records and reproducibility', '',
        f'- [2D GCNN study and complete protocol]({REPOSITORY}pythonTest/2D/docs/GCNN_STUDY.md)',
        f'- [Machine-readable 2D results]({REPOSITORY}pythonTest/2D/docs/gcnn-study-results.json)',
        f'- [Machine-readable 1D results]({REPOSITORY}docs/benchmarks/2026-10-30/results.json)', '',
        'Detailed local checkpoints, raw chains and historical sources are kept',
        'separately from the website. Fidelity and energy refer to the declared',
        'finite lattice and occupation cutoff. Compare [the conventions](physics.md)',
        'before interpreting results across dimensions.', '',
    ]
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='fail if the committed page differs from the records')
    args = parser.parse_args()
    expected = render()
    if args.check:
        if not DESTINATION.exists() or DESTINATION.read_text() != expected:
            parser.error('benchmark page is stale; run python docs/tools/prepare_reference.py')
        print('Benchmark page matches retained 1D and 2D records')
    else:
        DESTINATION.write_text(expected)
        print(DESTINATION)


if __name__ == '__main__':
    main()
