# Development and documentation

## Runtime checks

The existing Python checks use small systems and generated fixtures:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -B -m unittest discover -s pythonTest/2D/tests
.venv/bin/python pythonTest/vmc_gpu_test.py --self-test --device cpu
```

The Julia bridge tests cover legacy scalar calls, batched inference, 2D metadata
and geometry, import isolation, and the existing Python workflow wrappers.
They generate small checkpoints during testing rather than requiring ignored
research weights. Optional CUDA checks run only when explicitly enabled on a
machine with a working CUDA PyTorch environment.

```bash
.venv/bin/python -B -m unittest discover -s tests
./julia-local test/runtests.jl
BHNQS_TEST_CUDA=1 ./julia-local test/runtests.jl
```

Run the legacy 2D suite and the bridge suite in **separate Python processes**,
as above. They intentionally use different import routes; collecting both in
one interpreter would correctly trigger the duplicate-model identity guard.

## Build this site

The site uses Zensical's classic theme with ordinary Markdown sources in
`docs/site/`. Its environment contains only documentation dependencies:

```bash
python3 -m venv .venv-docs
.venv-docs/bin/python -m pip install -r docs/requirements-docs.txt
python3 docs/tools/prepare_reference.py
.venv-docs/bin/zensical build --strict
```

Preview with `.venv-docs/bin/zensical serve`. Generated output is in `site/` and
is ignored by Git. Physics equations use MathJax in the browser; the Python and
Julia code fences are syntax-highlighted without importing those libraries.

The benchmark page is generated from retained JSON records using Python's
standard library. No Julia, PyTorch, ED calculation, GPU or training is needed
to render the website.

## Continuous integration and Pages

Runtime tests and the website build are separate jobs. Pull requests build the
site and validate its links. Main-branch builds publish the site artifact to
GitHub Pages. Configure the repository's Pages publishing source as **GitHub
Actions** before the first deployment.

The Python/Julia CI jobs use a tested CPU environment. GPU checks remain an
optional local validation; research benchmarks are retained results and are
not retrained during documentation builds.

## Source and results

The [repository](https://github.com/angkun-research/BoseHubbardNQS) contains the
numerical source, tests, API references and compact benchmark reports. Large
checkpoints, raw chains and historical source snapshots are local artifacts,
not files served by this website. Reproducing a historical training run needs
its recorded environment and inputs; a seed alone does not promise identical
floating-point results across machines.

The [local integration verification record](https://github.com/angkun-research/BoseHubbardNQS/blob/main/docs/integration-verification.json)
records the tested runtime, CUDA checks, documentation checks and preservation
audit. It records local validation; GitHub-hosted CI and Pages deployment must
still complete after pushing.
