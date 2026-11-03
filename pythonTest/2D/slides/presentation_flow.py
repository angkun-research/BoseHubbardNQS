"""Presentation-only reordering of the already measured benchmark evidence."""
import copy


def equations(title, lines, footer, fontsize=37, step=.125):
    return dict(kind='equations', title=title, lines=lines, footer=footer,
                fontsize=fontsize, y=.80, step=step)


def measurement_appendix():
    return [
        equations('Exact measurement on small systems', [
            r'$p_N(\mathbf{n})=|\psi_N(\mathbf{n})|^2,\qquad \sum_{\mathbf{n}\in\mathcal{H}_N}p_N(\mathbf{n})=1$',
            r'$\rho_i=\sum_{\mathbf{n}\in\mathcal{H}_N}p_N(\mathbf{n})\,n_i$',
            r'$G_{ij}=\sum_{\mathbf{n},\mathbf{m}\in\mathcal{H}_N}\psi_N^*(\mathbf{n})\langle\mathbf{n}|b_i^\dagger b_j|\mathbf{m}\rangle\psi_N(\mathbf{m})$',
            'Enumerate all fixed-N configurations for BOTH ED and the saved NQS.',
            '2×3: 426 states; 3×3: 19,855 states (unit filling, cutoff4).'],
            'These are full-sector expectations without sampling error. Off-diagonal matrix elements connect occupations by one boson hop and include the bosonic square-root factor. Exact saved-NQS error versus ED measures the state approximation; it does not test the sampler.'),
        equations('Monte Carlo: density and coherence', [
            r'$\mathbf{n}^{(m)}\sim p_{\theta,N_0},\qquad \langle O\rangle\simeq M^{-1}\sum_m O_{\rm loc}(\mathbf{n}^{(m)})$',
            r'$O_{\rm loc}(\mathbf{n})=\sum_{\mathbf{m}}\langle\mathbf{n}|O|\mathbf{m}\rangle\,\psi_\theta(\mathbf{m})/\psi_\theta(\mathbf{n})$',
            r'$\rho_i\simeq M^{-1}\sum_m n_i^{(m)},\qquad D_i\simeq M^{-1}\sum_m n_i^{(m)}(n_i^{(m)}-1)$',
            r'$R_{ij}(\mathbf{n})=\sqrt{n_i(n_j+1)}\,e^{f_\theta(\mathbf{n}-\mathbf{e}_i+\mathbf{e}_j)-f_\theta(\mathbf{n})},\quad i\ne j$',
            r'$G_{ij}\simeq M^{-1}\sum_m R_{ij}(\mathbf{n}^{(m)}),\qquad G_{ii}=\rho_i$'],
            'Row-action convention: the sampled configuration is the bra; the connected ket removes one boson at i and adds it at j. Set R=0 unless n_i>0 and n_j<Nmax. Wavefunctions here are real and positive; each hop preserves total N. D_i is a factorial moment; Var(n_i)=D_i+rho_i−rho_i².', fontsize=35, step=.125),
        equations('Frozen measurements and error bars', [
            'Freeze the checkpoint; sample fixed-N chains with no parameter updates.',
            r'$g_{ij}^{(m)}=\frac{1}{2}(R_{ij}^{(m)}+R_{ji}^{(m)}),\qquad \overline{G}_\alpha^{(m)}=|B_\alpha(C)|^{-1}\sum_{\{i,j\}\in B_\alpha(C)}g_{ij}^{(m)}$',
            'Form each site/bond average PER SAMPLE, retaining its covariance.',
            'Block consecutive times within each walker before pooling block means.',
            r'$\mathrm{SEM}_{\rm shown}=\max(\mathrm{SEM}_{\rm blocks},\mathrm{SEM}_{\rm walker\ means})$'],
            'Frozen seeds321/654 are both retained; plots use predeclared654. Small2×3 cases:64walkers ×512retained steps, burn4096 proposals/walker, thin8, block64; new6×4 rows:128walkers, thin16. A proposal is one local move attempt, not a lattice sweep. SEM assumes stationarity, independent walkers and sufficiently decorrelated blocks; it is not proof of mixing. Larger NQS cylinders use MC versus DMRG, not enumeration.', fontsize=35, step=.13)
    ]


def reorganize(original, evidence):
    by_title = {s['title']: copy.deepcopy(s) for s in original}
    def get(title):
        return by_title[title]
    def section(title):
        return dict(kind='section', title=title)

    definitions = equations('Density and nearest-neighbor coherence', [
        r'$\rho_i=\langle n_i\rangle,\qquad G_{ij}=\langle b_i^\dagger b_j\rangle,\qquad g_{ij}=\operatorname{Re}G_{ij}$',
        'C is a chosen set of physical lattice sites; α denotes x or y.',
        r'$B_\alpha(C)=\{\{i,j\}\in E_\alpha:\ i\in C,\ j\in C\}$',
        'Eα contains unique unordered nearest-neighbor bonds along α, including wraps.',
        r'$\overline{G}_\alpha(C)=\frac{1}{|B_\alpha(C)|}\sum_{\{i,j\}\in B_\alpha(C)}g_{ij}$',
        'Main3×3 torus: C = all nine sites; |Bx(C)| = |By(C)| = 9.'],
        'A bond belongs to Bα(C) only when BOTH endpoints lie in C. Each physical edge is counted once. Coherence is a one-body correlation, not <n_i n_j>. At fixed N, <b_i>=0 and sum_i rho_i=N; uniform density alone does not establish superfluidity.', fontsize=36, step=.105)
    ed = evidence['observables']['square']['ED']
    visual = dict(kind='ed_state', title='ED ground state: density and coherence',
        Lx=3, Ly=3, density=ed['density'], rho=ed['rho'], Gx=ed['Gx'], Gy=ed['Gy'],
        footer='Actual3×3-torus ED ground state: N=9, cutoff4, t=1, U=16.74. Full19,855-state sums; site i=yLx+x. Translation symmetry gives rho_i=1; numerical roundoff is not a density texture. The right panel is the complete one-body matrix, with diagonal Gii=rho_i. Nearest-neighbor averages include all nine bonds per direction; nonzero coherence alone does not establish thermodynamic order.')

    density = get('Density benchmarks: NQS versus ED')
    coherence = get('Nearest-neighbor benchmarks')
    supervised_plot = dict(kind='plots', title='Supervised states: density and coherence',
        panels=[density['panels'][0], coherence['panels'][0]],
        footer='3×3 torus, N=9, cutoff4. Full fixed-sector enumeration for ED and each supervised NQS: no Monte Carlo. Left: all nine site densities minus ED. Right: all18 unique NN bond coherences minus ED, ordered x then y with wraps included. These residuals measure state approximation error.')
    vmc_plot = dict(kind='plots', title='VMC states: density and coherence',
        panels=[density['panels'][1], coherence['panels'][1]],
        footer='2×3 torus, N=6, cutoff4. Exact saved-NQS expectations use all426 configurations; frozenMC654 crosses show independent sampled estimates ± conditionalSEM. ED is also enumerated exactly. Both seeds appear in the appendix. Three x and six y bonds: this short rectangle need not have Gx=Gy.')
    gaps = get('These finite clusters have a gap')
    gaps['title'] = 'Neutral gap: excitation at fixed N'
    gaps['subtitle'] = r'$\Delta_N=E_1(N)-E_0(N)$: lowest two eigenvalues of the SAME fixed-N Hamiltonian.'
    gaps['footer'] = ('Neutral means no particle is added or removed: E1(N) is the first excited energy in the same sector as E0(N). '
        'This is an ED spectral gap, not a neural-network metric. The charge gap E0(N+1)+E0(N−1)−2E0(N) is different. '
        'Both eigenpairs are checked by residuals; these finite clusters do not establish an infinite-system gap.')

    leakage_definition = equations('Fixed-N fit and untrained particle sectors', [
        r'$\widetilde{\psi}_\theta(\mathbf{n})=e^{f_\theta(\mathbf{n})},\qquad Z_N=\sum_{\sum_i n_i=N}e^{2f_\theta(\mathbf{n})}$',
        r'$\psi_{\theta,N_0}(\mathbf{n})=\delta_{\sum_i n_i,N_0}\,e^{f_\theta(\mathbf{n})}/\sqrt{Z_{N_0}}$',
        r'$P_{\rm full}(N)=Z_N/\sum_{K=0}^{36}Z_K,\qquad \ell_{\rm full}=1-P_{\rm full}(N_0)$',
        r'$W=\{7,8,9,10,11\},\qquad P_W(N)=Z_N/\sum_{K\in W}Z_K$',
        'Training sees only N0=9; the raw network also returns values outside it.'],
        'Nine sites with local occupations0,…,4 allow N=0,…,36. Full probabilities normalize over all5^9 configurations; PW conditions only on W. The physical canonical state is zero outside N0; the raw exp(f) generally is not. Raw leakage tests an untrained extension, not the particle numbers visited by canonical VMC.', fontsize=36, step=.13)
    raw = get('Trained at N = 9: raw vs projected')
    nearby = get('Nearby untrained sectors: two diagnostics')
    raw['title'] = 'Measured raw sector weights'
    raw['panels'][1] = copy.deepcopy(nearby['panels'][0])
    raw['footer'] = ('Same supervised3×3-torus primary fits, trained ONLY at N=9. All5^9 configurations are enumerated. '
        'LEFT: raw globally normalized probabilities, with dots at N=9 and tails below10^-9 clipped. '
        'RIGHT: the same raw weights conditioned on N=7,…,11. High canonical overlap need not imply large raw mass at N=9; these are different normalizations.')
    controls = get('Same canonical state, different leakage')
    controls['rows'] = [r for r in controls['rows'] if r[0] != 'Exact number projection']
    projection = get('Projection is an exact control')
    projection['title'] = 'Enforce the intended physical support'
    projection['lines'][1] = r'$P^{\rm projected}(N)=\delta_{N,N_0},\quad \ell^{\rm projected}_{\rm full}=0,\quad F^{\rm projected}=F_{N_0}$'

    main = [get('Bose–Hubbard NQS\nPeriodic 2D systems'),get('What are we testing?'),
        section('1. Exact-diagonalization\nreference'),get('Hamiltonian'),get('Independent ED and DMRG check'),definitions,visual,
        section('2. Supervised NQS'),get('Five basic ansatzes'),get('Supervised objective'),get('Can basic NQS fit the ED state?'),get('Supervised fits: local observables'),supervised_plot,
        section('3. Small-system VMC'),get('Variational Monte Carlo'),get('Small-system VMC versus ED'),get('VMC trajectory and independent checks'),get('Frozen sampler versus exact state'),vmc_plot,get('Criticality and finite-size validation'),gaps,
        section('4. Particle-sector leakage'),leakage_definition,raw,get('Leakage and within-sector accuracy'),get('Fixed-sector data cannot fix leakage'),controls,projection,get('What this benchmark establishes')]

    detail = equations('Sector normalizations and overlap', [
        r'$\ell_{\rm near}=\sum_{N\in W,\ N\ne N_0}P_{\rm full}(N),\qquad \ell_{\rm full}=1-P_{\rm full}(N_0)$',
        r'$d_N=\dim\mathcal{H}_N,\qquad \overline{w}_N=Z_N/d_N,\qquad R_N=Z_N/Z_{N_0}$',
        r'$F_{\rm full}=\sqrt{P_{\rm full}(N_0)}\,F_{N_0}$',
        'Ffull compares the raw full-space state to zero-embedded canonical ED.'],
        'All overlaps are absolute and unsquared. Full means the finite cutoff0..4 basis; it does not mean unbounded Fock space. PW is conditional on the nearby window and cannot substitute for Pfull in either leakage definition.')
    # Preserve the established supporting calculations and hybrid measurements.
    appendix = [get('Supporting calculations'),get('Validate the ED construction'),*measurement_appendix(),detail,nearby]
    appendix += [copy.deepcopy(s) for s in original[29:] if s['title'] != 'Observable estimators in VMC']
    return main + appendix, len(main)
