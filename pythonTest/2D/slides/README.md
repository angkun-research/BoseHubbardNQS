# 2D benchmark slides

## Square-symmetry teaching deck

The new dedicated presentation is
[`BoseNQS2D-symmetry-progress-2026-11-02.pdf`](../BoseNQS2D-symmetry-progress-2026-11-02.pdf).
Its 58 pages follow four sections: theoretical operators, ED symmetry checks, explicit
group-CNN construction, and Adam + log-MSE results. It includes title-only
section dividers, site/filter diagrams, layer equations, training curves,
orbit-amplitude checks, exact observables and a mathematical appendix. The
accepted 55-page research presentation remains separate and unchanged.

The teaching revision defines the irreps through explicit matrices for the
generators R and M and translation phases, explains A1 projection weight, and
adds CNN/lifting/group-convolution schematics with all channel and orientation
indices defined. The main body has 46 pages, followed by 12 appendix pages;
proofs, scope notes and finite-size caveats are in the appendix.
[Page guide](symmetry-page-guide.json) lists the revised order;
[comment-to-page mapping](symmetry-revision-map.json) records every requested
change. Numerical evidence and model checkpoints are unchanged.

Rebuild this reviewed snapshot without checkpoint files or a GPU:

```bash
MPLCONFIGDIR=/tmp/bhnqs-slide-mpl .venv/bin/python -B pythonTest/2D/slides/build_symmetry_slides.py
```

`symmetry-evidence.json` freezes compact reviewed metrics, the 338 orbit
predictions, training snapshots and archival input hashes. `symmetry-deck.json`
is the canonical, directly editable slide source. Its evidence hash must match
the frozen numerical snapshot before rendering. The renderer writes a PDF hash and hashes of
all current presentation/code inputs in the adjacent provenance JSON. Archival
trainer/checkpoint hashes bind the evidence to the independently reviewed run.
The progress label describes a finite proof-of-principle study, with residual
errors and no universal expressiveness or larger-system VMC claim.

To recollect evidence after an independently reviewed new run, use
`collect_symmetry_evidence.py --trainer <retained-trainer.json> --review
<independent-review.json>`. It requires `approved=true` and matching
`trainer_report_sha256` and `checkpoint_sha256`. Recollection needs the local
archival reports; rendering the frozen snapshot does not. A new evidence snapshot
requires updating and reviewing the deck's numerical content and evidence hash;
the renderer never silently replaces results. `symmetry_diagrams.py` contains
only the custom diagrams, with no superseded slide text or replacement rules.
The reusable numerical
runners are `benchmarks/check_square_symmetry.py`,
`benchmarks/check_square_symmetry_independent.py` and
`benchmarks/train_square_group.py` (the latter creates fresh outputs and refuses
to overwrite retained checkpoints). No training occurs during slide rendering.

## Accepted research deck

The current research story is ED correctness, supervised representation,
small-system stochastic VMC and raw neighboring-sector amplitudes. Main-body
data are fully periodic in BOTH directions; cylinders are supporting context. The
physics-focused assembler below supersedes the larger-system common-accuracy
story as the main presentation; that older assembler remains available for
reproducing its historical comparison snapshot.

```bash
.venv/bin/python -B pythonTest/2D/slides/collect_physics_evidence.py
.venv/bin/python -B pythonTest/2D/slides/assemble_physics.py
MPLCONFIGDIR=/tmp/bhnqs-slide-mpl .venv/bin/python -B pythonTest/2D/slides/build_slides.py --deck slides/physics-deck.json --output BoseNQS2D-progress-2026-10-31.pdf
```

The collector requires independent approval of periodic leakage, periodic
DMRG/ED references, site/bond observable benchmarks and small-VMC evidence.
It derives numeric extracts from immutable run reports; it does not train or
evaluate a model. `physics_context.json` supplies the projection, normalization
and finite-gap definitions; `literature-context.json` records primary sources
and distinguishes literature statements from derivations made here. The
renderer records PDF and input hashes. Prior reviewed snapshots are retained
under ignored `outputs/slides-snapshots/`.

The requested November 1 backup extension uses `backup_extension.py` and the
reviewed records under `benchmarks/results/2026-11-01-backup-extension/`.
It fills the Jastrow/RBM cylinder rows and appends the 2D convolution plus
Transformer benchmarks. The November 1 snapshot preserved28 main slides; the
November 2 presentation revision reorganizes them into four sections. The
collector requires final numerical approval, retains failed first attempts and
selects only the explicitly predeclared conservative retries. It never chooses
the better frozen seed. The original five models remain the default sweep.

All leakage probabilities refer to a stated local cutoff. Full normalization
uses all37 particle sectors of the3×3 cutoff4 space; window probabilities
condition on N=7,…,11. Nearby leakage in the table is the *global* mass at
N=7,8,10,11, while total leakage includes every N≠9. Particle projection is an
algebraic control with no fitting. Density/NN illustrations use independent
3×3 TORUS ED data and all nine sites (nine x and nine y bonds), rather than the central strip used for larger
cylinder summaries.

The new deck follows `BoseNQS05-29-2026.pdf`, `BoseNQS06-05-2026.pdf`,
`BoseNQS07-07-2026.pdf` and `BoseNQS07-31-2026.pdf` in the parent directory:
1920x1080 point pages, white backgrounds, large black Helvetica-like titles,
STIX mathematics, gray-headed comparison tables, simple scientific plots,
architecture explanations, and final takeaway/to-do slides.

The user has intentionally removed those four style-template PDFs. Their
historical hashes are retained in `retired-style-references.json`; they are not
required for future rendering. The renderer records their retirement and still
rejects any other missing source input. The accepted PDF/provenance remains a
historical snapshot and is not rewritten by this housekeeping change.

Historical architecture-comparison sequence: title; Hamiltonian/lattice and references; canonical-sector
definition; four adapted networks and Transformer; ED-supervised comparisons;
VMC and SR definitions; Conv refinement; five-model energy comparisons; density
and one-body correlations; finite-size/reference checks; takeaways; to-do list.

Numerical slides must be generated from selected, audited result artifacts.
Distinguish absolute overlap from squared overlap, primary log-MSE fits from
optional infidelity refinements, and conditional sampling SEM from finite-chi
reference sensitivity. Larger-system ground-state fidelity is not measured.
Do not show a numerical target as achieved unless its recorded fit supports it.

`build_slides.py` renders the finalized declarative deck and records its input
hashes. Source and intermediate figures remain here; the final new PDF goes in
the parent `2D/` directory. The original PDFs are read-only references.

Regenerate the data manifest and PDF from the repository root:

```bash
MPLCONFIGDIR=/tmp/bhnqs-slide-mpl .venv/bin/python -B pythonTest/2D/slides/assemble_results.py --selection slides/selection.json
MPLCONFIGDIR=/tmp/bhnqs-slide-mpl .venv/bin/python -B pythonTest/2D/slides/build_slides.py --deck slides/deck.json
```

`selection.json` records the selected assessment for each model and cylinder.
The assembler checks model/geometry metadata and requires evaluations with seeds
321 and 654. Numerical tables always use the predeclared independent confirmation
seed 654; their summary gate requires both seeds to meet the recorded point and
precision criteria. No favorable-seed selection or silent averaging is used.
Until all selected comparisons are audited, the deck is marked partial and the
final renderer refuses to publish it. Review-only previews live under ignored
`outputs/slides-style/`.

For a checked intermediate research snapshot, use `assemble_results.py --progress`
with a separate audited selection manifest and output deck. This keeps missing
comparisons explicit, labels the evaluation seed on every energy row, and uses
the title “2D benchmark progress.” The renderer accepts that snapshot only with
an output filename containing `-progress-`; it cannot replace the final deck.
The snapshot's audit concerns the reported evidence, not achievement of every
research target.

The historical architecture-comparison deck includes an eight-page mathematical appendix
(`ridge_sr_appendix.json`, pages 29–36): canonical normalization and score
centering; imaginary time and the quantum metric; the ridge least-squares
objective and native-SGD update; the exact smaller-space solve with `M*tau`;
eigenmode damping; the published MinSR soft filter; complexity/precision; and
the difference from the original 1D convention. The main result selection now
uses basic displacement-shared Jastrow as a fixed baseline. Boundary-orbit
experiments remain archived and are rejected by the main-deck assembler.

```bash
MPLCONFIGDIR=/tmp/bhnqs-slide-mpl .venv/bin/python -B pythonTest/2D/slides/assemble_results.py --progress --selection slides/progress-selection.json --output slides/progress-deck.json
MPLCONFIGDIR=/tmp/bhnqs-slide-mpl .venv/bin/python -B pythonTest/2D/slides/build_slides.py --deck slides/progress-deck.json --output BoseNQS2D-progress-2026-10-31.pdf
```

In that historical deck, the first matched RBM64 optimizer protocol/results are on
pages 38–39. Both evaluation seeds are shown. Main result tables retain ridge
as the default; they do not select the smaller MinSR mean. Four neural families
at both sizes define the completed comparison dataset; Jastrow is a fixed
physical baseline and can be reported without passing that gate.

In the historical comparison deck, page 13 is a small-system, small-sample one-step comparison using production
solvers (`small_system_minsr.json`). Pages 14–15 give the density and NN
one-body definitions and lattice visualizations; page 37 supplies their Monte
Carlo estimators (`observable_definitions.json`). Density colors use actual
DMRG values. The NN graph highlights the four x bonds and eight y bonds,
including periodic wraps, whose endpoints both lie in the central strip.
The current deck has29main pages and55total. Title-only section dividers are
3,8,14,22. ED:4–7, including combined observable definitions6 and the actual ED
density/one-body matrix7. Supervision:9–13. VMC:15–21, with the fixed-N neutral
gap explicitly defined on21. Leakage:23–28, in the order definitions, measured
weights, quantitative comparison, nonidentifiability, numerical control,
projection. Page29 gives the takeaways.

Appendix: ED construction checks31; exact and Monte Carlo measurements32–34;
normalization/mean-weight details35–36; ridge mathematics37–44; MinSR45;
Transformer46; cylinder tables47–48; both-seed VMC observables49; hybrid50–55.
`presentation_flow.py` performs this presentation-only rearrangement after the
audited evidence is assembled. `page-guide.json` records the current order;
`progress-quality-check.json` records hashes and layout checks. Prior58-page
PDF and source snapshots remain archived under their PDF hash.

Slide24 compares raw global probabilities (log scale, training-sector markers)
with nearby-window probabilities. Exact number projection closes the main
leakage section on28; the mean-weight diagnostic is supporting detail on36.
Evaluating raw inputs at N=0,…,36 is a deliberate off-sector probe; it is not
the distribution of particle numbers used in training or canonical VMC.
