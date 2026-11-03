"""Collect the reviewed physics evidence; no fitting, sampling or model changes."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
FOCUS='benchmarks/results/2026-11-01-physics-focus/'
PERIODIC='benchmarks/results/2026-11-01-periodic-focus/'
NAMES={'FCNet':'FC','Conv2DNet':'Conv2D','RBMNet':'RBM','TransformerNet':'Transformer','JastrowNet':'Jastrow'}


def main():
    sources=['slides/collect_physics_evidence.py']
    def read(name):
        path=(ROOT/name).resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError('Evidence must remain under2D')
        sources.append(str(path.relative_to(ROOT)))
        return json.loads(path.read_text())
    for name in (FOCUS+'checker-small-vmc-review.json',
                 PERIODIC+'checker-periodic-leakage-results-review.json',
                 PERIODIC+'checker-periodic-reference-review.json',
                 PERIODIC+'checker-periodic-observable-review.json'):
        review=read(name)
        if not review['status'].startswith('APPROVED'):
            raise ValueError(f'{name} has not passed independent review')
    for name in ('checker-evidence-inventory.json','manager-ed-tests.json',
                 'checker-rbm-within-family-number-gauge.json'):
        read(FOCUS+name)
    read(PERIODIC+'primary-checkpoint-selection.json')
    read(PERIODIC+'manager-periodic-single-particle.json')
    small={r['arch']:r for r in read('benchmarks/results/2026-10-30-ed-2x3-w32-e10000/summary.json')['runs']}
    small['RBMNet']=next(r for r in read('benchmarks/results/2026-10-30-ed-2x3-RBM32-e20000/summary.json')['runs'] if r['seed']==123)
    leak=read(PERIODIC+'periodic-leakage-results.json')
    assert leak['status']=='completed' and leak['full_status']=='completed_all_37_sectors'
    assert leak['settings']['pbc_x'] and leak['settings']['pbc_y']
    supervised=[]; leakage=[]
    for r in leak['models']:
        sectors=r['full']['sectors']
        assert [z['N'] for z in sectors]==list(range(37))
        assert sum(z['dimension'] for z in sectors)==5**9
        probabilities=[z['P_full'] for z in sectors]
        assert abs(sum(probabilities)-1)<1e-10
        canonical=r['canonical']
        supervised.append(dict(model=NAMES[r['family']],
            small_overlap=small[r['family']]['best_checkpoint_metrics']['amplitude_overlap_abs_unsquared'],
            square_overlap=canonical['F_N0'],square_energy_error_per_site=canonical['energy_error_to_ed']/9,
            square_variance=canonical['variance']))
        leakage.append(dict(model=NAMES[r['family']],overlap=canonical['F_N0'],full_probability=probabilities,
            window_probability=[z['window_conditional_probability'] for z in r['nearby']['sectors']],
            logZ=[z['logZ_N'] for z in sectors],dimensions=[z['dimension'] for z in sectors]))
    vmc=read(FOCUS+'small-vmc-proof-results.json')
    assert vmc['status']=='completed'
    models=[]
    for r in vmc['models']:
        assert r['status']=='completed' and len(r['trajectory_MC'])==500
        final=r['exact_final']; frozen=[]
        for seed in ('321','654'):
            q=r['frozen'][seed]
            assert q['status']=='completed'
            report=read(q['report'])
            assert report['checkpoint_sha256']==r['checkpoint_sha256']
            comparison=q['comparison_to_exact']
            frozen.append(dict(seed=int(seed),energy=comparison['MC_energy_mean'],sem=comparison['conditional_energy_SEM']))
        models.append(dict(model=NAMES[r['family']],initial_energy=r['exact_initial']['energy'],
            energy=final['energy'],overlap=final['F_N0'],variance=final['variance'],frozen=frozen,
            updates=[z['epoch'] for z in r['trajectory_MC']],
            sampled_energy_per_site=[z['energy_per_site'] for z in r['trajectory_MC']]))
    gaps=read(FOCUS+'checker-neutral-gap.json')
    gaprows=[dict(geometry=f"{z['settings']['Lx']}×{z['settings']['Ly']} torus / N={z['settings']['Ntot']}",
                  energy=z['lowest_two_energies'][0],gap=z['neutral_gap_Delta_N']) for z in gaps['records']
             if z['settings']['pbc_x'] and z['settings']['pbc_y']]
    ed=read(PERIODIC+'checker-3x3-torus-independent-ed.json')
    assert ed['pbc_x'] and ed['pbc_y']
    gaprows.append(dict(geometry='3×3 torus / N=9',energy=ed['energy'],gap=ed['neutral_gap']))
    dmrg=read(PERIODIC+'checker-dmrg-torus-validation.json')
    assert dmrg['settings']['pbc_x'] and dmrg['settings']['pbc_y']
    primary,refined=dmrg['phases']
    checks=[['Basis dimension / unique bonds',f"{ed['dimension']:,} / {len(ed['bonds'])}"],
            ['Independent ED ground energy / t',f"{ed['energy']:.12f}"],
            ['Independent ED residual norm',f"{ed['residual_norm']:.3g} t"],
            ['DMRG energy error: max bond128',f"{primary['ED_differences']['energy_abs']:.3g} t"],
            ['DMRG energy error: max bond256',f"{refined['ED_differences']['energy_abs']:.3g} t"],
            ['Max full one-body error: max bond256',f"{refined['ED_differences']['rho_full_max_abs']:.3g}"],
            ['Max site-density error: max bond256',f"{refined['ED_differences']['density_max_abs']:.3g}"]]
    obs=read(PERIODIC+'periodic-observable-benchmarks.json')
    assert obs['status']=='completed' and obs['input_hashes_unchanged']
    datasets={z['id']:z for z in obs['datasets']}
    normalized={}
    for key,identity in (('square','supervised_3x3_torus'),('small','vmc_2x3_torus')):
        data=datasets[identity]
        assert data['geometry']['pbc_x'] and data['geometry']['pbc_y']
        normalized[key]=dict(ED=data['reference'],models=[],geometry=data['geometry'])
        for r in data['models']:
            origin=leak['models'] if key=='square' else vmc['models']
            expected=next(z for z in origin if z['family']==r['family'])
            assert r['checkpoint_sha256']==expected['checkpoint_sha256']
            row=dict(model=NAMES[r['family']],**r['exact']) if key=='square' else dict(model=NAMES[r['family']],exact=r['exact'],frozen=[])
            if key=='small':
                for q in r['frozen']:
                    values=q['observed'];statistics=q['statistics']
                    frozen=dict(seed=q['seed'],density=values['density'],density_sem=statistics['density']['conditional_sem'])
                    for name in ('meanD','Gx','Gy'):
                        frozen[name]=values[name];frozen[name+'_sem']=statistics[name]['conditional_sem']
                    row['frozen'].append(frozen)
                assert [q['seed'] for q in row['frozen']]==[321,654]
            normalized[key]['models'].append(row)
    sources+=list(obs['immutable_input_sha256'])
    sources+=['leakage_analysis.py','benchmarks/run_leakage_analysis.py','benchmarks/run_small_vmc_proof.py',
              'benchmarks/periodic_observables.py','benchmarks/run_periodic_benchmarks.py','benchmarks/check_torus_ed.py','dmrg_reference.jl']
    evidence=dict(audited=True,main_geometry='periodic_both_axes',scope='Fully periodic finite-size physics, measured observables and raw-sector extension; cylinders are supporting context.',
        sources=list(dict.fromkeys(sources)),ed_checks=checks,supervised=supervised,neutral_gaps=gaprows,
        small_vmc=dict(ed_energy=vmc['ED_reference']['energy'],models=models),leakage=leakage,observables=normalized)
    output=ROOT/'slides/physics-evidence.json'
    output.write_text(json.dumps(evidence,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(output)


if __name__=='__main__':
    main()
