"""Ensure the curated public Julia API covers every exported symbol."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
source = (ROOT / 'src/BoseHubbardNQS.jl').read_text()
document = (ROOT / 'docs/site/api/julia.md').read_text()
exports = []
for line in source.splitlines():
    match = re.match(r'^export\s+(.+)$', line)
    if match:
        exports.extend(name.strip() for name in match[1].split(','))
missing = [name for name in exports if not re.search(r'(?<!\w)' + re.escape(name) + r'(?!\w)', document)]
if missing:
    raise SystemExit(f'Public API is missing from docs/site/api/julia.md: {missing}')
print(f'All {len(exports)} exported Julia symbols are documented')
