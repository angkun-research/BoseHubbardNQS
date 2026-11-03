"""Build the physics-focused deck from explicitly audited evidence.

The input manifest stores checked numeric extracts and names every source
artifact. The original architecture/optimization deck remains reproducible.
"""
import copy
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COLORS = {'FC': '#3575a2', 'Jastrow': '#bb7937', 'Conv2D': '#3d9367',
          'RBM': '#a25283', 'Transformer': '#7562aa'}


def read(name):
    return json.loads((ROOT / name).read_text())


def table(title, columns, rows, footer='', **kwargs):
    return dict(kind='table', title=title, columns=columns, rows=rows,
                footer=footer, fontsize=32, **kwargs)


def curve(label, x, y, **kwargs):
    result=dict(label=label,x=x,y=y,color=COLORS.get(label,'black'))
    result.update(kwargs)
    return result


def observable_benchmarks(data):
    square,small=data['square'],data['small']
    reference=square['ED']
    rows=[['ED','0',f"{reference['meanD']:.6f}",f"{reference['Gx']:.6f}",f"{reference['Gy']:.6f}"]]
    density_curves=[];bond_curves=[]
    for r in square['models']:
        drho=[v-w for v,w in zip(r['density'],reference['density'])]
        dg=[v-w for v,w in zip(r['bond_G'],reference['bond_G'])]
        rows.append([r['model'],f'{max(abs(v) for v in drho):.3g}',f"{r['meanD']:.6f}",f"{r['Gx']:.6f}",f"{r['Gy']:.6f}"])
        density_curves.append(curve(r['model'],list(range(9)),drho,marker='o'))
        bond_curves.append(curve(r['model'],list(range(18)),dg,marker='o'))
    output=[table('Supervised fits: local observables', ['3×3 torus','max |δρᵢ|','Mean D','Gx','Gy'],rows,
        'Exact CANONICAL ED and NQS expectations, not MC. Density is compared SITE BY SITE; its global mean is fixed to1. D=<n(n−1)> is an onsite factorial moment. Gx/Gy average all nine unique bonds per direction, including periodic wraps.',widths=[.22,.23,.19,.18,.18])]
    small_density=[]
    small_G=[curve('ED',[0,1],[small['ED']['Gx'],small['ED']['Gy']],linestyle='--')]
    for r in small['models']:
        exact=r['exact'];mc=next(q for q in r['frozen'] if q['seed']==654)
        small_density.append(curve(r['model']+' exact',list(range(6)),
            [v-w for v,w in zip(exact['density'],small['ED']['density'])],color=COLORS[r['model']],marker='o'))
        small_density.append(curve(r['model']+' MC654',list(range(6)),
            [v-w for v,w in zip(mc['density'],small['ED']['density'])],color=COLORS[r['model']],marker='x',linestyle='None',yerr=mc['density_sem']))
        small_G.append(curve(r['model']+' exact',[0,1],[exact['Gx'],exact['Gy']],color=COLORS[r['model']],marker='o'))
        small_G.append(curve(r['model']+' MC654',[0,1],[mc['Gx'],mc['Gy']],color=COLORS[r['model']],marker='x',linestyle='None',yerr=[mc['Gx_sem'],mc['Gy_sem']]))
    output.append(dict(kind='plots',title='Density benchmarks: NQS versus ED',panels=[
        dict(title='3×3 torus: supervised fits',xlabel='Physical site i',ylabel='Density difference from ED',
             series=density_curves,xticks=list(range(9)),reference_lines=[dict(y=0,label='ED')],legend_size=18,legend_ncol=3,legend_loc='upper center'),
        dict(title='2×3 torus: VMC states',xlabel='Physical site i',ylabel='Density difference from ED',
             series=small_density,xticks=list(range(6)),reference_lines=[dict(y=0,label='ED')],legend_size=21)],
        footer='Both systems are fully periodic; observables use FIXED-N states and site index i=yLx+x. ED density is1 at every site. LEFT: exact learned-state expectations. RIGHT: exact VMC curves and frozenMC654 ± conditionalSEM; no favorable-seed selection. Conserved total N does not force a variational profile to be uniform.'))
    output.append(dict(kind='plots',title='Nearest-neighbor benchmarks',panels=[
        dict(title='3×3 torus: supervised fits',xlabel='Bond index (0–8: x; 9–17: y)',ylabel='Bond coherence difference from ED',
             series=bond_curves,xticks=[0,4,8,12,17],
             vertical_lines=[dict(x=8.5)],reference_lines=[dict(y=0,label='ED')],legend_size=18,legend_ncol=3,legend_loc='upper right'),
        dict(title='2×3 torus: VMC states',xlabel='Bond direction',ylabel='Mean NN coherence',series=small_G,
             xticks=[0,1],xticklabels=['x (3 bonds)','y (6 bonds)'],legend_size=21)],
        footer='LEFT: all18unique3×3 bonds, exact NQS minus ED; wraps included. RIGHT: ED, exact saved VMC states and frozenMC654 ± conditionalSEM. Hermitian bonds and direction averages are formed within each raw sample before blocking, retaining covariance. Rectangular2×3 need not have Gx=Gy.'))
    return output


def observable_MC_table(data):
    small=data['small'];ref=small['ED']
    rows=[['ED',f"{ref['meanD']:.6f}",f"{ref['Gx']:.6f}",f"{ref['Gy']:.6f}"]]
    for r in small['models']:
        exact=r['exact'];rows.append([r['model']+' exact',*[f"{exact[k]:.6f}" for k in ('meanD','Gx','Gy')]])
        for q in r['frozen']:
            rows.append([f"{r['model']} MC{q['seed']}",*[f"{q[k]:.5f} ± {q[k+'_sem']:.5f}" for k in ('meanD','Gx','Gy')]])
    return table('Periodic VMC: local observables',['2×3 torus','Mean D','Gx','Gy'],rows,
        'BOTH frozen seeds321/654 retained; exact refers to full426-state expectation of that NQS, while ED is the reference ground state. D=<n(n−1)>; sample composites precede blocking. Errors are conditionalSEM, not confidence intervals or mixing proofs.',widths=[.25,.25,.25,.25])


def main():
    evidence = read('slides/physics-evidence.json')
    if evidence.get('audited') is not True:
        raise ValueError('Physics evidence must pass independent review first')
    if evidence.get('main_geometry') != 'periodic_both_axes':
        raise ValueError('Main-body evidence must be periodic in both directions')
    context = read('slides/physics_context.json')
    old = read('slides/progress-deck.json')
    by_title = {s['title']: s for s in old['slides']}
    base = read('slides/base_slides.json')
    base_by_title = {s['title']: s for s in base['slides']}
    slides = [dict(kind='title', title='Bose–Hubbard NQS\nPeriodic 2D systems',
                   subtitle='ED construction • supervised representation • small-system VMC',
                   date='10-31-2026'), context['purpose'], copy.deepcopy(base_by_title['Hamiltonian'])]
    slides[-1].update(Lx=3,Ly=3,pbc_x=True,pbc_y=True)
    slides[-1]['footer'] = ('Main-body systems are periodic in BOTH directions; the diagram is a3×3 torus. '
        'Cylinder results remain in supporting material. All hopping bonds are unique unordered pairs; short length-two axes have one bond per pair.')
    slides[-1]['lines'][-1] = 'Periodic in both x and y (torus)'
    slides.append(table('Validate the ED construction', ['Independent check', 'What it establishes'], [
        ['Atomic limit: t = 0', 'U n(n−1)/2 diagonal and fixed-N chemical shift'],
        ['One particle,3×3 torus', 'E0/t = −4 (uniform degree-four hopping graph)'],
        ['Two particles / two sites', 'E0 = (U−√(U²+16t²))/2 (one hopping bond)'],
        ['Explicit ladder operators', 'Independent 2×2 Hamiltonian and bosonic factors'],
        ['Dense versus sparse ED', 'Small-system spectrum and eigenvector residual'],
        ['Number and boundaries', 'Canonical basis, wraps, no self/doubled bonds']],
        '15 ED-specific tests pass; the periodic single-particle result is checked separately. Independent operator construction tests the physics conventions; solver residual alone would not validate the Hamiltonian.', widths=[.39,.61]))
    checks = evidence['ed_checks']
    slides.append(table('Independent ED and DMRG check', ['Quantity', '3×3 torus'],
        [[k,v] for k,v in checks],
        'Both axes periodic; N=9, cutoff4, t=1, U=16.74. DMRG uses a periodic Hamiltonian with an open-chain MPS representation. Requested maximum bond dimensions128 and256 are checked against ED; the physical boundaries are never opened.', widths=[.44,.56]))
    architectures=copy.deepcopy(base_by_title['Neural networks'])
    architectures['title']='Five basic ansatzes'
    architectures['items']=[
        dict(name='FC',layers=['One-hot input','Tanh dense layers','Scalar log amplitude'],note='No imposed lattice symmetry.'),
        dict(name='Conv2D',layers=['One-hot grid','3×3 Conv + SiLU','Dense scalar head'],note='Boundary-aware padding. Global dense head.'),
        dict(name='Jastrow',layers=['Onsite tables','Shared pair tables','Sum log amplitude'],note='Fixed physical baseline.'),
        dict(name='RBM',layers=['One-hot input','Softplus hidden sum','Bias + scalar'],note='Binary hidden units.'),
        dict(name='Transformer',layers=['Occupation + (x,y)','One attention block','Mean pool + scalar'],note='ReLU, one head; no dropout.')]
    objective=copy.deepcopy(base_by_title['Supervised objective'])
    objective['footer']='Main fits use Adam + log-MSE over the chosen particle sector only. Reported overlaps and energies are checked against the raw ED ground vector. Unseen particle-number sectors never enter this loss.'
    slides += [architectures,objective]
    slides.append(table('Can basic NQS fit the ED state?', ['Ansatz', '2×3 Fabs', '3×3 Fabs', '3×3 δE/site'],
        [[r['model'],f"{r['small_overlap']:.6f}",f"{r['square_overlap']:.6f}",f"{r['square_energy_error_per_site']:.4g}"] for r in evidence['supervised']],
        'ALL results are periodic in both axes. Adam + normalized log-MSE; checkpoint selected by log-MSE. Four neural fits reach Fabs≥0.9999 on2×3 and≥0.99 on3×3. Width32, actual trained primary fits. RBM tiny fit20000updates, others10000; Jastrow is a physical baseline.', widths=[.24,.25,.25,.26]))
    slides.append(copy.deepcopy(base_by_title['Variational Monte Carlo']))
    vmc = evidence['small_vmc']
    slides.append(table('Small-system VMC versus ED', ['Ansatz', 'Initial E/t', 'Final E/t', 'Final Fabs', 'Var(H)/t²'],
        [[r['model'],f"{r['initial_energy']:.4f}",f"{r['energy']:.6f}",f"{r['overlap']:.6f}",f"{r['variance']:.4g}"] for r in vmc['models']],
        f"2×3 torus, N=6, cutoff4, 426 states. ED E/t={vmc['ed_energy']:.9f}. Fresh initialization;500 stochastic SR+nativeSGD updates, no ED labels or supervised warm start. Exact final-state metrics; variance remains nonzero.", widths=[.18,.22,.22,.20,.18]))
    panels=[]
    for r in vmc['models']:
        panels.append(dict(title=r['model'], xlabel='Stochastic update', ylabel='Energy per site / t',
            series=[curve(r['model'],r['updates'],r['sampled_energy_per_site'])],
            reference_lines=[dict(y=vmc['ed_energy']/6,label='ED')], legend_size=23))
    slides.append(dict(kind='plots', title='VMC trajectory and independent checks', panels=panels,
        footer='Curves are noisy training Monte Carlo estimates before each update, not exact energies or monotone descent. Exact final metrics and frozen321/654 measurements are assessed separately. One fit per model is a proof of principle, not an architecture ranking.'))
    slides.append(table('Frozen sampler versus exact state', ['Ansatz / seed', 'MC E/t ± conditional SEM', 'Exact saved-state E/t'],
        [[f"{r['model']} / {q['seed']}",f"{q['energy']:.5f} ± {q['sem']:.5f}",f"{r['energy']:.6f}"]
         for r in vmc['models'] for q in r['frozen']],
        'Both evaluation seeds are retained; neither selects the training checkpoint. Error estimates are conditional on mixing and stationarity, not confidence intervals or proof of equilibration.'))
    ed_observables=read('benchmarks/results/2026-11-01-periodic-focus/checker-3x3-torus-independent-ed.json')
    density=copy.deepcopy(by_title['Density on the lattice'])
    coherence=copy.deepcopy(by_title['Nearest-neighbor correlations'])
    for diagram in (density,coherence):
        diagram.update(Lx=3,Ly=3,pbc_x=True,pbc_y=True,central_columns=[0,1,2],bonds=ed_observables['bonds'],
                       central_bond_counts={'x':9,'y':9},region_label='C = all nine sites')
    density.update(density=ed_observables['density'],density_reference_label='ED, 3×3 torus')
    density['lines']=[line.replace('|C|=8','|C|=9').replace('N_{\\rm tot}=16','N_{\\rm tot}=9') for line in density['lines']]
    density['lines'][0]=r'$\rho_i=\langle n_i\rangle=\sum_{\mathbf n\in\mathcal{H}_{N_0}}p_{N_0}(\mathbf n)\,n_i$'.replace(r'\mathbf n',r'\mathbf{n}')
    density['footer']='Independent3×3-torus ED at N=9 has uniform mean density1 by translation symmetry. ED roundoff is not physical density variation. Pointwise ED-versus-NQS comparisons follow; average density alone does not test correlations.'
    coherence['footer']='One-body coherence, not density-density correlation. The3×3 torus has nine x bonds and nine y bonds, including three wraps along each axis. All are used in the benchmarks. Nonzero NN coherence alone does not establish superfluidity or criticality.'
    slides += [density,coherence]
    slides += observable_benchmarks(evidence['observables'])
    slides.append(context['criticality'])
    slides.append(table('These finite clusters have a gap', ['Geometry / sector', 'E0/t', 'Neutral gap ΔN/t'],
        [[r['geometry'],f"{r['energy']:.6f}",f"{r['gap']:.6f}"] for r in evidence['neutral_gaps']],
        'Lowest two eigenpairs checked by residuals; tiny case also checked by dense diagonalization. This fixed-N neutral gap differs from E0(N+1)+E0(N−1)−2E0(N). No finite-size scaling or infinite-system gap is inferred.'))
    slides += [context['raw_projected'], context['leakage_definition']]
    leak = evidence['leakage']
    full_curves=[]; window_curves=[]; mean_curves=[]
    window=list(range(7,12))
    for r in leak:
        full_curves.append(curve(r['model'],list(range(37)),r['full_probability']))
        full_curves.append(dict(x=[9],y=[r['full_probability'][9]],color=COLORS[r['model']],marker='o',linestyle='None'))
        window_curves.append(curve(r['model'],window,r['window_probability'],marker='o'))
        reference=r['logZ'][9]-math.log(r['dimensions'][9])
        mean_curves.append(curve(r['model'],window,
            [r['logZ'][n]-math.log(r['dimensions'][n])-reference for n in window],marker='o'))
    slides.append(dict(kind='plots',title='Trained at N = 9: raw vs projected',panels=[
        dict(title='Raw network, all-sector normalization',xlabel='Tested N (training N = 9)',ylabel='Raw Pfull(N), log scale',
             series=full_curves,legend_size=22,yscale='log',ylim=[1e-9,1],xlim=[0,36],xticks=[0,9,18,27,36],
             vertical_lines=[dict(x=9)]),
        dict(title='Projected physical state',xlabel='Total particle number N',ylabel='Pprojected(N)',
             series=[dict(label='Every projected model',x=window,y=[0,0,1,0,0],color='black',marker='o',linestyle='--')],
             xticks=window,ylim=[-.03,1.08],legend_size=22)],
        footer='Same3×3-TORUS primary fits; training and canonical VMC use ONLY N=9. LEFT tests raw untrained sectors: nine sites × cutoff4 allow N=0,…,36. Dots mark Pfull(9); tails below10^-9 are clipped. RIGHT imposes exact number projection: P(9)=1 and every other sector has zero weight.'))
    slides.append(table('Leakage and within-sector accuracy', ['Ansatz', 'F9', 'Pfull(9)', 'Nearby leakage', 'Total leakage'],
        [[r['model'],f"{r['overlap']:.6f}",f"{r['full_probability'][9]:.4g}",
          f"{100*sum(r['full_probability'][n] for n in (7,8,10,11)):.4f}%",f"{100*(1-r['full_probability'][9]):.6f}%"] for r in leak],
        '3×3 TORUS, cutoff4. Nearby leakage sums GLOBAL probabilities of N=7,8,10,11; total leakage sums every N≠9. These are not window-conditional probabilities. High canonical overlap can coexist with large raw leakage; no universal architecture ranking follows.', widths=[.20,.18,.18,.22,.22]))
    slides.append(dict(kind='plots',title='Nearby untrained sectors: two diagnostics',panels=[
        dict(title='Raw, conditional on N = 7,…,11',xlabel='Tested particle number N',ylabel='Window-conditional PW(N)',
             series=window_curves,xticks=window,legend_size=22,vertical_lines=[dict(x=9)]),
        dict(title='Mean weight per configuration',xlabel='Total particle number N',
            ylabel='log(w̄N / w̄9)',series=mean_curves,xticks=window,legend_size=23,
            reference_lines=[dict(y=0,label='Equal mean weight')])],
        footer='Both panels use UNPROJECTED3×3-torus outputs. LEFT renormalizes over N=7,…,11 only, not the full space or the physical fixed-N state. RIGHT removes sector multiplicity using w̄N=ZN/dN; d7=6,030 and d11=48,879. Dividing by w̄9 removes the overall amplitude scale.'))
    slides.append(context['nonidentifiability'])
    control=next(r for r in leak if r['model']=='FC')
    controls=[]
    for label,tilt in [('Raw output',lambda n:0.),
                       ('h = −0.5(N−9)',lambda n:-.5*(n-9)),
                       ('h = +0.5(N−9)',lambda n:.5*(n-9)),
                       ('h = −(N−9)²',lambda n:-(n-9)**2)]:
        weights=[z+2*tilt(n) for n,z in enumerate(control['logZ'])]
        largest=max(weights)
        lognorm=largest+math.log(sum(math.exp(z-largest) for z in weights))
        probability=math.exp(weights[9]-lognorm)
        controls.append([label,f'{probability:.5g}',f"{control['overlap']:.6f}",'unchanged'])
    controls.append(['Exact number projection','1',f"{control['overlap']:.6f}",'unchanged'])
    slides.append(table('Same canonical state, different leakage', ['Postprocessing of FC log amplitude','Pfull(9)','F9','Canonical energy'],controls,
        'Algebraic reweighting of the SAME measured logZN, without new inference, fitting or checkpoint selection. h(9)=0 throughout. Changes in raw full-space weight cannot be detected by any fixed-N9 fitting metric.',widths=[.43,.19,.19,.19]))
    slides.append(context['projection'])
    slides.append(dict(kind='text',title='What this benchmark establishes',fontsize=41,step=.14,lines=[
        'ED: independent physics and numerical checks agree.',
        'Supervision: basic neural ansatzes approximate the ED state.',
        'VMC: validate stochastic optimization on small finite clusters.',
        'Raw neighboring-sector amplitudes are a separate diagnostic.',
        'Exact number projection enforces the intended physical support.'],
        footer='Large-system critical performance and improved NQS optimization remain separate questions. Neither leakage nor a finite-budget VMC plateau by itself proves an expressiveness limit.'))
    main_pages=len(slides)
    slides.append(dict(kind='title',title='Supporting calculations',subtitle='SR derivation, sampling checks and cylinder context',date='Appendix'))
    slides += read('slides/ridge_sr_appendix.json')['slides']
    transformer=copy.deepcopy(base_by_title['Simplest Transformer'])
    for item in transformer['items']:
        if 'code' in item:
            item['code']=item['code'].replace('d_model=64','d_model=32').replace('dim_feedforward=128','dim_feedforward=64')
        item['note']=item.get('note','').replace('Width 64 used for cylinder qualification.','Width32 matches the periodic primary fits.')
    slides += [copy.deepcopy(by_title['Small-system SR comparison']), copy.deepcopy(by_title['Observable estimators in VMC']),
               transformer,
               copy.deepcopy(by_title['Compare: 4×4 cylinder']), copy.deepcopy(by_title['Compare: 6×4 cylinder'])]
    slides.append(observable_MC_table(evidence['observables']))
    sources=['slides/assemble_physics.py','slides/physics-evidence.json','slides/physics_context.json',
             'slides/literature-context.json','slides/progress-deck.json','slides/base_slides.json','slides/ridge_sr_appendix.json']
    sources += evidence['sources']+old['sources']
    extension = ROOT/'benchmarks/results/2026-11-01-backup-extension/backup-extension-results.json'
    if extension.exists():
        from backup_extension import build
        slides[41], supporting, extension_sources = build(slides[41], evidence)
        slides.extend(supporting)
        sources.extend(extension_sources)
    from presentation_flow import reorganize
    slides, main_pages = reorganize(slides, evidence)
    sources.append('slides/presentation_flow.py')
    deck=dict(status='audited_progress_snapshot',title='Bose–Hubbard NQS: periodic 2D benchmarks and sector leakage',
              sources=list(dict.fromkeys(sources)),main_pages=main_pages,slides=slides)
    (ROOT/'slides/physics-deck.json').write_text(json.dumps(deck,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(f'{main_pages} main pages, {len(slides)} total')


if __name__=='__main__':
    main()
