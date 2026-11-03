# Current project plan

## New bounded study: group-equivariant NQS

Current phase is the user-authorized **3×3, N=9 symmetry and supervised GCNN
study**. Identify microscopic symmetry operators; verify the ED state; construct
and explain a regular-representation group CNN; fit with Adam + normalized
log-MSE; prepare a separate detailed teaching PDF covering these four steps.
No VMC or larger-size campaign is included.

The user now authorizes a new architecture study on square fully periodic
systems, using ED-first and Adam + log-MSE. This supersedes the earlier pause
on new architecture experiments, while Julia integration remains unmerged.

First identify the microscopic group and ground-state irrep; see
[docs/SQUARE_SYMMETRY.md](docs/SQUARE_SYMMETRY.md). The target is fixed charge
N0=L², zero momentum, and A1 of D4. Verify this numerically against the existing
independent3×3 ED reference. Build an equivariant hidden representation
with an invariant scalar readout and qualify it against raw ED. Review exact
equivariance and the fixed training protocol before sustained training. Keep the
existing architectures, accepted results,1D code and VMC implementation intact.
No VMC or larger-size extension precedes supervised qualification.

Initial ED-only results: all72 geometric operations preserve the3×3 ground
state to2.2e-15; the19,855 configurations form338 spatial orbits. The zero-
momentum/A1 sector has dimension338, despite A1 itself being a1D irrep.
Symmetry fixes equality within an orbit; it leaves337 independent real state
parameters after normalization/positivity/global-phase conventions. The
338×338 reduced Hamiltonian reproduces the full ED vector and energy.

## Preparation before Julia integration

The declared group-CNN fit and independent numerical review are complete
(absolute overlap 0.9996088830; energy error/site 0.001840578116t). The dedicated
58-page [teaching deck](BoseNQS2D-symmetry-progress-2026-11-02.pdf) covers all four
requested stages and retains the finite-fit errors and comparison limitations.
No additional fit, VMC, larger lattice or Julia merge is part of this task.

No merge, root package edit or1D functionality change is authorized by this
preparation step. Preserve the accepted presentation, numerical results,
checkpoint semantics and current Python defaults.

1. Document the current2D public functions, returns, shapes, failure modes and
   device/precision choices in [docs/API.md](docs/API.md).
2. Keep one shared PyTorch VMC core with tested derivative/solver options;
   do not introduce a duplicate Julia or NumPy VMC implementation.
3. Remove disposable caches/previews, archive unneeded scratch material and
   superseded prose, and preserve every benchmark/provenance dependency.
4. Review the existing PythonCall/1D adapter and explicitly document geometry,
   interaction, indexing, checkpoint and raw-log-amplitude differences in
   [docs/JULIA_PREPARATION.md](docs/JULIA_PREPARATION.md).
5. Verify the documentation examples, import/CLI entry points, CPU regression
   tests, protected inputs and local links after cleanup.
6. Stop at a reviewable handoff. A future user-authorized merge should first add
   collision-free Python namespaces and metadata-dispatched inference, then
   batch observables/VMC, followed by Julia↔Python parity checks. No such adapter
   or Julia integration test is claimed in this phase.

The completed research scope and known misses are in [STATUS.md](STATUS.md).
The previous broad refinement/large-system campaign remains closed; the new
bounded square-symmetry study above is a separate user-requested direction.
Historical plans remain under `docs/history/pre-documentation-cleanup/` for
traceability; they do not override this scope.
