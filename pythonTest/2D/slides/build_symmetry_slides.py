"""Build the separate graduate introduction to the square-torus GCNN study.

Uses the existing white 16:9 presentation style. Does not replace the accepted
55-page research presentation. Run from the repository root with .venv Python.
The reviewed deck JSON is the canonical presentation source. Its numerical
evidence hash must match before rendering; rendering never assembles or trains models.
"""
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_slides as style
import symmetry_diagrams as diagrams

ROOT = style.ROOT
BLUE = "#286b9b"


def panel_lattice(ax, values, title, labels=True, vmin=0, vmax=4, cmap="Blues"):
    values = np.asarray(values).reshape(3, 3)
    ax.imshow(values, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax,
              extent=(-.5, 2.5, -.5, 2.5))
    ax.set_xticks(range(3), ["0", "1", "2"])
    ax.set_yticks(range(3), ["0", "1", "2"])
    ax.set_xlabel("x", fontsize=28)
    ax.set_ylabel("y", fontsize=28)
    ax.set_title(title, fontsize=32, pad=18)
    if labels:
        for y in range(3):
            for x in range(3):
                val = values[y, x]
                ax.text(x, y, f"{val:g}", ha="center", va="center", fontsize=38,
                        color="white" if val > (vmin+vmax)/2 else "black")
    for v in [.5, 1.5]:
        ax.axhline(v, color="white", lw=3)
        ax.axvline(v, color="white", lw=3)


def draw_custom(fig, slide):
    kind = slide["kind"]
    if kind == "site_maps":
        original = np.arange(9)
        for k, (name, rotation, mirror, dx, dy) in enumerate([
                ("Site labels", 0, False, 0, 0), ("Translation $T_x$", 0, False, 1, 0),
                ("Quarter-turn $R$", 1, False, 0, 0), ("Mirror $M$", 0, True, 0, 0)]):
            values = np.empty(9, dtype=int)
            for i in range(9):
                x, y = i % 3, i // 3
                if mirror:
                    y = -y
                for _ in range(rotation):
                    x, y = -y, x
                values[(x+dx)%3 + 3*((y+dy)%3)] = original[i]
            ax = fig.add_axes([.055 + .24*k, .29, .18, .36])
            panel_lattice(ax, values, name, vmax=8, cmap="YlGnBu")
        fig.text(.07, .13, "Numbers label sites, not boson occupations. Every coordinate is modulo 3.", fontsize=34)
    elif kind == "orbit_examples":
        for k, name in enumerate(("nearest_neighbor", "diagonal")):
            data = slide["examples"][name]
            ax = fig.add_axes([.08 + .49*k, .30, .26, .39])
            panel_lattice(ax, data["configuration"], "Nearest-neighbor pair" if k == 0 else "Diagonal pair")
            fig.text(.08 + .49*k, .19, f"36 configurations in this orbit\nED amplitude = {data['amplitude']:.7f}", fontsize=32)
        fig.text(.07, .09, f"Same onsite interaction energy; different orbit. Amplitude ratio = {slide['ratio']:.5f}.", fontsize=34)
    elif kind == "filter_bank":
        kernel = np.array([[0, 1, 0], [2, 0, 0], [0, 0, 3]])
        for mirror in (0, 1):
            for rotation in range(4):
                # origin='lower': positive physical R(x,y)=(-y,x) requires
                # negative NumPy array quarter-turns (rows increase upward).
                values = np.rot90(np.flip(kernel, axis=0) if mirror else kernel, -rotation)
                ax = fig.add_axes([.085 + .23*rotation, .48-.28*mirror, .16, .21])
                ax.imshow(values, cmap="Blues", origin="lower", vmin=0, vmax=3)
                ax.set_xticks([]); ax.set_yticks([])
                for y in range(3):
                    for x in range(3):
                        ax.text(x, y, str(values[y,x]), ha="center", va="center", fontsize=24,
                                color="white" if values[y,x] > 1 else "black")
                ax.set_title(f"$R^{rotation}$" + ("$M$" if mirror else ""), fontsize=30)
        fig.text(.07, .09, "One learned filter, eight tied copies. The feature maps keep all eight responses.", fontsize=34)
    elif kind == "ed_heatmaps":
        for k, (values, name, vmax) in enumerate([(slide["density"], "$\\langle n_i\\rangle$", 1.2),
                (slide["one_body_from_zero"], "$\\langle b_0^\\dagger b_j\\rangle$", 1.)]):
            ax = fig.add_axes([.11+.47*k,.28,.28,.44])
            panel_lattice(ax, np.round(values, 4), name, labels=False, vmax=vmax)
            for j, value in enumerate(values):
                ax.text(j%3,j//3,f"{value:.3f}",ha="center",va="center",fontsize=29,
                        color="white" if value > vmax/2 else "black")
        fig.text(.07,.13,"Translation fixes the density; symmetry relates directions, not correlation magnitudes.",fontsize=34)
    elif kind == "scatter":
        ax = fig.add_axes([.105,.20,.39,.55])
        ax.scatter(slide["ed"],slide["nqs"],s=13,alpha=.6,color=BLUE)
        limits=[min(slide["ed"]),max(slide["ed"])]
        ax.plot(limits,limits,"--",color="black",lw=2)
        ax.set(xlabel="ED log amplitude",ylabel="GCNN log amplitude")
        ax.grid(alpha=.2)
        ax = fig.add_axes([.60,.20,.32,.55])
        ax.hist(np.asarray(slide["nqs"])-np.asarray(slide["ed"]),bins=35,color=BLUE,edgecolor="white")
        ax.set(xlabel="Log-amplitude error",ylabel="Number of orbits")
        ax.grid(alpha=.2)
    else:
        raise ValueError(f"unknown custom slide {kind}")


def main():
    deck_path = ROOT / "slides/symmetry-deck.json"
    evidence_path = ROOT / "slides/symmetry-evidence.json"
    deck = json.loads(deck_path.read_text())
    if deck["status"] != "audited_progress_snapshot":
        raise ValueError("require a reviewed progress deck")
    if deck["numerical_evidence_sha256"] != hashlib.sha256(evidence_path.read_bytes()).hexdigest():
        raise ValueError("evidence changed; update and review the deck before rendering")
    slides = deck["slides"]
    output = ROOT / "BoseNQS2D-symmetry-progress-2026-11-02.pdf"
    with tempfile.TemporaryDirectory(dir=ROOT, prefix=".symmetry-slides-stage-") as temporary:
        staged = Path(temporary) / "slides.pdf"
        with PdfPages(staged, metadata={"Title": deck["title"], "Author": "Bose-Hubbard NQS project"}) as pdf:
            for number, slide in enumerate(slides, 1):
                if slide["kind"] in diagrams.CUSTOM:
                    fig = style.page(slide["title"], slide.get("footer", ""), number)
                    diagrams.draw(fig, slide)
                elif slide["kind"] in {"site_maps", "orbit_examples", "filter_bank", "ed_heatmaps", "scatter"}:
                    fig = style.page(slide["title"], slide.get("footer", ""), number)
                    draw_custom(fig, slide)
                else:
                    fig = style.render(slide, number)
                pdf.savefig(fig)
                plt.close(fig)
        staged.replace(output)
    sources = [Path(__file__), Path(style.__file__), Path(diagrams.__file__), deck_path, evidence_path]
    sources += [ROOT / name for name in (
        "benchmarks/check_square_symmetry.py", "docs/SQUARE_SYMMETRY.md",
        "slides/collect_symmetry_evidence.py", "models.py", "square_group.py",
        "benchmarks/train_square_group.py", "benchmarks/check_square_symmetry_independent.py")]
    provenance = {"status": deck["status"], "pages": len(slides), "page_points": [1920, 1080],
                  "sources_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                     for path in sources},
                  "pdf_sha256": hashlib.sha256(output.read_bytes()).hexdigest()}
    output.with_suffix(".provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"{output}: {len(slides)} pages")


if __name__ == "__main__":
    main()
