# Git contents and local research artifacts

The large filesystem size is mostly generated research data, not source code.
Keep these separate when committing:

| Content | Git policy |
| --- | --- |
| Python/Julia source, tests, documentation, slide sources and final PDF | Include |
| `outputs/` — checkpoints, raw MC arrays, DMRG outputs, archived PDFs | Local only; ignored |
| `benchmarks/results/` — dated reports, trajectories, logs and experiment evidence | Local only; ignored |
| Slide previews and Python caches | Ignore; regenerate as needed |

Adding an ignore rule does not unstage files that were already added. The
previously staged generated `benchmarks/results/` files are removed from the
index, while their local files remain intact. No history rewrite, numerical-data
deletion, commit, push or1D/2D merge is performed.

[local-artifacts.json](local-artifacts.json) records relative artifact paths,
byte counts and SHA256 hashes. This small manifest is not a substitute for the
data. Keep a separate backup: Git does not back up ignored files. To move
artifacts to another machine, restore both directories at the same paths under
`pythonTest/2D/`. No external storage service/download is configured here.

The standard CPU suite uses temporary synthetic fixtures and saved models.
Three optional archival-forward regression checks skip when historical
checkpoints/gold records are absent; they still validate hashes and forwards
when those local artifacts are present. Ordinary model, ED, sampler, SR and
observable tests do not require the research archive.

The rendered PDFs and compact slide manifests can be read from Git. The group-CNN
teaching presentation renders directly from its reviewed deck and compact
evidence JSON; it does not need checkpoint files or the local artifact bundle.
Recollecting its evidence, regenerating the earlier full research presentation,
running archival comparisons, and the
documented existing-checkpoint API example require the local artifact bundle.
The four user-retired style templates remain optional. Large cylinder raw
chains have also been intentionally retired by the user, with original hashes
and retained reports listed in [retired-mc-chains.json](../slides/retired-mc-chains.json).
All six small2×3 validation chains remain locally available, as do every
checkpoint, report and log. The earlier inventory is a historical snapshot;
local deletion receipts live under `outputs/cleanup-plans/`.

Cached slide expectations and reported cylinder summaries remain available.
Future rendering records the retired raw-chain inputs explicitly. Reanalysing
those cylinder chains (different blocking, mixing checks or new observables)
requires fresh sampling; saved summaries do not restore the deleted samples.
No other missing input is silently accepted.
