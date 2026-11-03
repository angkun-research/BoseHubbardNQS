# Preparation before Julia integration

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
There is no new training, architecture search or larger-system campaign.
Historical plans remain under `docs/history/pre-documentation-cleanup/` for
traceability; they do not override this scope.
