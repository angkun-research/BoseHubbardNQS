"""Prepare backup slides from the reviewed, bounded November 1 extension."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = 'benchmarks/results/2026-11-01-backup-extension/'


def build(cylinder, primary):
    from assemble_physics import table, curve

    sources = ['slides/backup_extension.py', 'slides/hybrid_architecture.json']

    def source(name):
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            raise ValueError(f'Missing or external backup evidence: {name}')
        sources.append(str(path.relative_to(ROOT)))
        return path

    def read(name):
        return json.loads(source(name).read_text())

    review = read(RESULTS + 'checker-backup-extension-results-review.json')
    if not review['status'].startswith('APPROVED'):
        raise ValueError('Backup numerical evidence has not passed independent review')
    result = read(RESULTS + 'backup-extension-results.json')
    if result['status'] not in ('completed', 'completed_with_execution_failures'):
        raise ValueError('Backup runs have unresolved execution failures')
    protocol = read(RESULTS + 'backup-extension-protocol.json')
    read(RESULTS + 'checker-backup-extension-readiness.json')
    read(RESULTS + 'checker-backup-extension-partial-ed-review.json')
    read(RESULTS + 'checker-backup-hybrid-final-review.json')
    read(RESULTS + 'manager-backup-physics-review.json')
    read(RESULTS + 'checker-jastrow6-data-review.json')
    read(RESULTS + 'checker-rbm6-data-review.json')
    read(RESULTS + 'backup-extension-final-handoff.json')
    read(RESULTS + 'pre-extension-baseline-gold.json')
    source(RESULTS + 'pre-extension-baseline-gold.npz')
    sources.extend(protocol['source_sha256'])
    retries_used = []
    for family, row_index, label in (('JastrowNet', 1, 'Jastrow'), ('RBMNet', 3, 'RBM128')):
        model = result['cylinders'][family]
        if model['status'] != 'completed':
            prefix, checker = (('Jastrow-basic-6x4', 'jastrow6') if family == 'JastrowNet'
                               else ('RBM128-6x4', 'rbm6'))
            model = read(RESULTS + prefix + '-stable-retry-results.json')
            retry = read(RESULTS + prefix + '-stable-retry-protocol.json')
            read(RESULTS + 'checker-' + checker + '-retry-readiness.json')
            source(retry['failed_predecessor']['log'])
            sources.extend(retry['source_sha256'])
            if model['checkpoint'] != retry['checkpoint']:
                raise ValueError('Cylinder result must be the single predeclared retry')
            label += ' retry'
            retries_used.append(label)
        if model['status'] != 'completed':
            raise ValueError(f'Incomplete cylinder result: {family}')
        source(model['checkpoint']); read(model['trainer_report'])
        assessment = read(model['assessment'])
        reports = assessment['reports']
        if len(reports) != 2:
            raise ValueError('Both frozen evaluation seeds are required')
        for q in model['frozen'].values():
            read(q['report']); source(q['raw_samples'])
        final = next(q for q in reports if q['report'].endswith('seed654.json'))
        stat = final['statistics']['whole_lattice_energy_per_site']
        gap = abs(stat['mean'] - stat['reference']) / abs(stat['reference'])
        passed = all(q['all_four_summary_point_and_precision_targets_met'] for q in reports)
        gate = 'baseline' if family == 'JastrowNet' else ('pass' if passed else 'below target')
        cylinder['rows'][row_index] = [label + '\nseed 654', str(model['parameter_count']),
            f"{stat['mean']:.6f} ± {stat['conditional_sem']:.6f}", f'{100*gap:.2f}%', gate]
    cylinder['highlight_rows'] = [i + 1 for i, row in enumerate(cylinder['rows']) if row[-1] == 'below target']
    cylinder['footer'] = ('6×4 CYLINDER: open x, periodic y; N=24, cutoff4, U/t=16.74. '
        'All rows use frozen seed654; gap is relative energy/site error to the same chi128 DMRG reference. '
        'Neural pass requires BOTH seeds: whole E/site and central-strip density/Gx/Gy point AND precision targets. '
        'Jastrow is the original displacement baseline, exempt from the neural gate. Budgets differ; no architecture ranking.')
    if retries_used:
        cylinder['footer'] += ' J/RBM first attempts became nonfinite; selected fresh retries use smaller steps and fixed larger ridge (protocols retained).'

    hybrid = result['hybrid']
    slides = [read('slides/hybrid_architecture.json')]
    supervised_rows = []
    for geometry in ('2x3', '3x3'):
        row = hybrid['supervised'][geometry]
        metrics = row['independent_CPUfloat64']
        source(row['checkpoint']); read(row['trainer_report'])
        n = 6 if geometry == '2x3' else 9
        supervised_rows.append([geometry.replace('x', '×') + ' torus',
            str(metrics['parameter_count']), f"{metrics['F_N0']:.10f}",
            f"{metrics['energy_error_to_ed']/n:.4g}", 'pass' if row['admission_passed'] else 'miss'])
    slides.append(table('Hybrid: supervised ED benchmarks',
        ['System', 'Parameters', 'Fabs', 'δE/site / t', 'ED gate'], supervised_rows,
        'Width32; one four-head block. Adam + normalized log-MSE,10000 updates, seed123; saved by log-MSE. '
        'Raw-ED checks in CPUfloat64. Both axes periodic, cutoff4. Tiny gate: Fabs≥0.9999 and δE/site≤0.001t; '
        '3×3 working gate: Fabs≥0.99. A single fixed-budget fit tests this setup, not universal capacity.',
        widths=[.23,.18,.22,.22,.15]))

    vmc = hybrid['vmc']
    if vmc['status'] != 'completed':
        slides.append(dict(kind='text', title='Hybrid: VMC admission', lines=[vmc['reason']], fontsize=37,
            footer='No VMC run is claimed when the predeclared supervised admission is missed.'))
    else:
        final = vmc['exact_final']; ref = final['ED']; obs = final['observables']
        source(vmc['checkpoint']); source(vmc['initial_checkpoint']); read(vmc['trainer_report'])
        rows = []
        for r in primary['small_vmc']['models']:
            rows.append([r['model'], f"{r['energy']:.7f}", f"{r['overlap']:.7f}", f"{r['variance']:.4g}"])
        rows.append(['Conv + Transformer', f"{final['energy']:.7f}", f"{final['F_N0']:.7f}", f"{final['variance']:.4g}"])
        slides.append(table('Hybrid: fresh small-system VMC', ['2×3 torus', 'Exact E/t', 'Exact Fabs', 'Var(H)/t²'], rows,
            f"ED E/t={ref['energy']:.9f}. Fresh seed123;500 stochastic ridge-SR + nativeSGD updates, M=1024; no ED labels or warm start. "
            'Exact full426-state final expectations. Hybrid runs GPUfloat32/TF32off with float64 SR; earlier FC/Conv run CPUfloat64. '
            'Different parameter counts and arithmetic: this is a functionality check, not a controlled ranking.', widths=[.31,.25,.24,.20]))
        rows = [['ED', f"{ref['energy']:.7f}", *[f"{ref[k]:.6f}" for k in ('meanD','Gx','Gy')]],
                ['Hybrid exact', f"{final['energy']:.7f}", *[f"{obs[k]:.6f}" for k in ('meanD','Gx','Gy')]]]
        for seed in ('321','654'):
            q = vmc['frozen'][seed]
            read(q['report']); source(q['raw_samples'])
            stats = q['observables']['statistics']
            rows.append(['Hybrid MC' + seed, *[f"{stats[k]['mean']:.5f} ± {stats[k]['conditional_sem']:.5f}" for k in ('energy','meanD','Gx','Gy')]])
        slides.append(table('Hybrid: frozen observable checks', ['2×3 torus', 'E/t', 'Mean D', 'Gx', 'Gy'], rows,
            'BOTH frozen evaluation seeds retained, CPUfloat64,64walkers ×512retained steps, block64. '
            'D=<n(n−1)>; Gx/Gy average the three/six unique NN bonds. Composite each sample BEFORE blocking to retain covariance. '
            'Errors are max(blockSEM,between-chainSEM), conditional on mixing; exact-state error and MC scatter are separate.',
            widths=[.23,.20,.19,.19,.19]))
        slides[-1]['fontsize'] = 26
        q = vmc['frozen']['654']['observables']
        slides.append(dict(kind='plots', title='Hybrid: density and NN coherence', panels=[
            dict(title='Site density relative to ED', xlabel='Physical site i', ylabel='Density difference', xticks=list(range(6)),
                series=[curve('Hybrid exact',list(range(6)),[a-b for a,b in zip(obs['density'],ref['density'])],color='#267a89',marker='o'),
                        curve('Hybrid MC654',list(range(6)),[a-b for a,b in zip(q['observed']['density'],ref['density'])],
                            color='#267a89',marker='x',linestyle='None',yerr=q['statistics']['density']['conditional_sem'])],
                reference_lines=[dict(y=0,label='ED')],legend_size=23),
            dict(title='Every unique NN bond',xlabel='Bond index (0–2: x; 3–8: y)',ylabel='Coherence difference from ED',xticks=list(range(9)),
                series=[curve('Hybrid exact',list(range(9)),[a-b for a,b in zip(obs['bond_G'],ref['bond_G'])],color='#267a89',marker='o'),
                        curve('Hybrid MC654',list(range(9)),[a-b for a,b in zip(q['observed']['bond_G'],ref['bond_G'])],
                            color='#267a89',marker='x',linestyle='None',yerr=q['statistics']['bond_G']['conditional_sem'])],
                vertical_lines=[dict(x=2.5)],reference_lines=[dict(y=0,label='ED')],legend_size=23)],
            footer='2×3 fully periodic, fixed N=6. Curves are exact expectations of the saved VMC state minus ED; crosses are frozenMC654 ± conditionalSEM. '
            'Includes periodic wraps and Hermitian bond estimators. Pointwise density, rather than its conserved mean, tests the learned profile.'))

    leak = read(hybrid['leakage'])
    r = leak['models'][0]
    if leak['full_status'] == 'completed_all_37_sectors':
        sectors = r['full']['sectors']; probabilities = [z['P_full'] for z in sectors]
        if len(sectors) != 37 or abs(sum(probabilities)-1) > 1e-10:
            raise ValueError('Full hybrid sector scan is incomplete')
        slides.append(dict(kind='plots',title='Hybrid: raw neighboring-sector weights',panels=[
            dict(title='Raw full-space normalization',xlabel='Total particle number N',ylabel='Pfull(N), log scale',
                series=[curve('Conv + Transformer',list(range(37)),probabilities,color='#267a89',marker='o')],
                yscale='log',ylim=[1e-9,1],xlim=[0,36],xticks=[0,9,18,27,36],vertical_lines=[dict(x=9)],legend_size=22),
            dict(title='Nearby conditional probabilities',xlabel='Total particle number N',ylabel='P(N | 7≤N≤11)',
                series=[curve('Raw hybrid',list(range(7,12)),[z['window_conditional_probability'] for z in r['nearby']['sectors']],color='#267a89',marker='o'),
                        curve('Exact projection',list(range(7,12)),[0,0,1,0,0],color='black',linestyle='--')],xticks=list(range(7,12)),legend_size=22)],
            footer=f"Same supervised3×3 torus checkpoint, trained ONLY at N=9: F9={r['canonical']['F_N0']:.6f}, raw Pfull(9)={probabilities[9]:.5g}. "
                'All5^9 cutoff configurations enumerated; left-panel tails below10^-9 are clipped. Right panel renormalizes ONLY over N=7,…,11. '
                'Raw leakage is unconstrained extension, not canonical VMC contamination; exact number projection enforces physical support without retraining.'))
    else:
        slides.append(dict(kind='text',title='Hybrid: sector scan status',fontsize=36,lines=[
            'Nearby sectors evaluated; full-space scan exceeded its budget.',
            'No global leakage probability is inferred from a window.'],
            footer='See the retained sector record for the measured window-conditional weights and full-scan status.'))
    return cylinder, slides, list(dict.fromkeys(sources))
