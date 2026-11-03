#!/usr/bin/env python3
"""Render a data-backed 16:9 research deck in the supplied 1D slide style.

Usage: .venv/bin/python -B pythonTest/2D/slides/build_slides.py --deck slides/deck.json
Paths are relative to pythonTest/2D/. No original PDF can be overwritten.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import tempfile
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.colors import TwoSlopeNorm

ROOT = Path(__file__).resolve().parents[1]
PAGE = (1920 / 72, 1080 / 72)
plt.rcParams.update({"font.family": ["Liberation Sans", "DejaVu Sans"], "mathtext.fontset": "stix",
                     "pdf.fonttype": 42, "font.size": 40, "axes.labelsize": 32,
                     "xtick.labelsize": 25, "ytick.labelsize": 25,
                     "legend.fontsize": 25, "axes.spines.top": False,
                     "axes.spines.right": False})


def resolve(value):
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("all slide paths must stay under pythonTest/2D/")
    return path


def page(title, footer, number):
    fig = plt.figure(figsize=PAGE, facecolor="white")
    fig.text(.045, .935, title, fontsize=84, weight="bold", va="top")
    if footer:
        fig.text(.045, .035, textwrap.fill(footer, 185), fontsize=18, color=".28", va="bottom")
    fig.text(.96, .035, str(number), fontsize=18, color=".45", ha="right")
    return fig


def text_lines(fig, lines, x=.07, y=.75, step=.14, fontsize=44):
    for line in lines:
        fig.text(x, y, line, fontsize=fontsize, va="top")
        y -= step


def table(fig, slide):
    ax = fig.add_axes(slide.get("box", [.07, .16, .86, .60]))
    ax.axis("off")
    tab = ax.table(cellText=slide["rows"], colLabels=slide["columns"],
                   cellLoc="center", loc="center", bbox=[0, 0, 1, 1],
                   colWidths=slide.get("widths"))
    tab.auto_set_font_size(False)
    tab.set_fontsize(slide.get("fontsize", 34))
    for (row, column), cell in tab.get_celld().items():
        cell.set_edgecolor(".60")
        cell.set_linewidth(1.3)
        cell.PAD = .08
        if row == 0:
            cell.set_facecolor("#d5dcdc")
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor("#f4f4f4" if row % 2 == 0 else "white")
        if row in slide.get("highlight_rows", []):
            cell.get_text().set_color("#a32020")
    if slide.get("subtitle"):
        fig.text(.07, .80, slide["subtitle"], fontsize=32, va="top")


def plot_panels(fig, slide):
    panels = slide["panels"]
    count = len(panels)
    for index, panel in enumerate(panels):
        width = .82 / count
        ax = fig.add_axes([.085 + index * (.87 / count), .19, width - .045, .55])
        for series in panel["series"]:
            opts = {"label": series.get("label"), "color": series.get("color"),
                    "linestyle": series.get("linestyle", "-"),
                    "marker": series.get("marker", None), "linewidth": 2.5}
            if "yerr" in series:
                ax.errorbar(series["x"], series["y"], yerr=series["yerr"],
                            capsize=5, markersize=8, **opts)
            else:
                ax.plot(series["x"], series["y"], **opts)
        for line in panel.get("reference_lines", []):
            ax.axhline(line["y"], color=line.get("color", "black"),
                       linestyle="--", linewidth=2, label=line.get("label"))
        for line in panel.get("vertical_lines", []):
            ax.axvline(line["x"], color=line.get("color", ".5"),
                       linestyle="--", linewidth=1.5, label=line.get("label"))
        ax.set(xlabel=panel.get("xlabel", ""), ylabel=panel.get("ylabel", ""))
        ax.set_title(panel.get("title", ""), fontsize=35, pad=22)
        if "ylim" in panel:
            ax.set_ylim(panel["ylim"])
        if "xlim" in panel:
            ax.set_xlim(panel["xlim"])
        if "xticks" in panel:
            ax.set_xticks(panel["xticks"], labels=panel.get("xticklabels"))
        if panel.get("yscale"):
            ax.set_yscale(panel["yscale"])
        ax.grid(alpha=.2)
        if panel.get("legend", True):
            ax.legend(frameon=False, fontsize=panel.get("legend_size", 24),
                      loc=panel.get("legend_loc", "best"), ncol=panel.get("legend_ncol", 1))


def lattice(fig, slide):
    ax = fig.add_axes([.58, .16, .35, .59])
    Lx, Ly = slide.get("Lx", 4), slide.get("Ly", 4)
    pbc_x, pbc_y = slide.get("pbc_x", False), slide.get("pbc_y", True)
    for x in range(Lx):
        for y in range(Ly):
            ax.plot(x, y, "o", color="black", markersize=13)
            if x + 1 < Lx:
                ax.plot([x, x + 1], [y, y], color=".35", lw=2)
            if y + 1 < Ly:
                ax.plot([x, x], [y, y + 1], color=".35", lw=2)
        if pbc_y and Ly > 2:
            ax.add_patch(FancyArrowPatch((x, 0), (x, Ly - 1), arrowstyle="-",
                                         connectionstyle="arc3,rad=-.30", color="#35688d",
                                         linestyle="--", linewidth=2))
    if pbc_x and Lx > 2:
        for y in range(Ly):
            ax.add_patch(FancyArrowPatch((0, y), (Lx - 1, y), arrowstyle="-",
                                         connectionstyle="arc3,rad=.30", color="#b54c4c",
                                         linestyle="--", linewidth=2))
    ax.set(xlim=(-.7, Lx + .3), ylim=(-.6, Ly - .3), aspect="equal")
    ax.axis("off")
    ax.text((Lx - 1) / 2, -.65, "$x$: " + ("periodic" if pbc_x else "open"), ha="center", fontsize=35)
    ax.text(Lx + .1, (Ly - 1) / 2, "$y$:\n" + ("periodic" if pbc_y else "open"), ha="left", fontsize=30)


def architecture(fig, slide):
    items = slide["items"]
    for index, item in enumerate(items):
        x = .065 + index * .90 / len(items)
        width = .83 / len(items)
        fig.text(x, .78, item["name"], fontsize=40, weight="bold")
        if "code" in item:
            ax = fig.add_axes([x, .23, width, .48])
            ax.set_facecolor("#252525")
            ax.set_xticks([]); ax.set_yticks([])
            ax.text(.035, .96, item["code"], transform=ax.transAxes,
                    color="#dce8df", fontfamily="DejaVu Sans Mono",
                    fontsize=item.get("code_size", 22), va="top", linespacing=1.5)
        else:
            for j, label in enumerate(item["layers"]):
                y = .65 - j * .10
                fig.add_artist(Rectangle((x, y), width, .065, transform=fig.transFigure,
                                         facecolor=".94", edgecolor=".55"))
                fig.text(x + width / 2, y + .032, label, ha="center", va="center", fontsize=27)
                if j + 1 < len(item["layers"]):
                    fig.text(x + width / 2, y - .017, "↓", ha="center", va="center", fontsize=27)
        if item.get("note"):
            limit = max(12, int(width * 1920 / (25 * .53)))
            note = "\n".join(textwrap.fill(line, limit) for line in item["note"].splitlines())
            fig.text(x, .15, note, fontsize=25, va="top")


def observable(fig, slide):
    """Physical lattice and central-strip averages; all values come from the deck."""
    text_lines(fig, slide["lines"], x=.055, y=.79, step=.145, fontsize=36)
    ax = fig.add_axes([.61, .25, .29, .48])
    Lx, Ly = slide["Lx"], slide["Ly"]
    columns = slide["central_columns"]
    ax.add_patch(Rectangle((columns[0]-.45, -.45), len(columns)-.1, Ly-.1,
                           facecolor="#fff4cf", edgecolor="#a88628", lw=1.5, zorder=0))
    for i, j in slide["bonds"]:
        x1,y1 = i % Lx,i // Lx
        x2,y2 = j % Lx,j // Lx
        direction = "x" if y1 == y2 else "y"
        central = x1 in columns and x2 in columns
        color = ("#c54545" if direction == "x" else "#356dc0") if central and slide["observable"] == "nn" else ".7"
        width = 4 if central and slide["observable"] == "nn" else 1.5
        if direction == "x" and abs(x1-x2)>1 and slide.get("pbc_x", False):
            ax.add_patch(FancyArrowPatch((min(x1,x2),y1),(max(x1,x2),y2), arrowstyle="-",
                connectionstyle="arc3,rad=.24", linestyle="--", color=color, lw=width, zorder=1))
        elif direction == "y" and abs(y1-y2)>1:
            ax.add_patch(FancyArrowPatch((x1,min(y1,y2)),(x2,max(y1,y2)), arrowstyle="-",
                connectionstyle="arc3,rad=-.24", linestyle="--", color=color, lw=width, zorder=1))
        else:
            ax.plot([x1,x2],[y1,y2],color=color,lw=width,zorder=1)
    xs=[i % Lx for i in range(Lx*Ly)]
    ys=[i // Lx for i in range(Lx*Ly)]
    if slide["observable"] == "density":
        values=slide["density"]
        span=max(max(values)-1,1-min(values),1e-4)+.0006
        dots=ax.scatter(xs,ys,c=values,cmap="coolwarm",norm=TwoSlopeNorm(vmin=1-span,vcenter=1,vmax=1+span),
                         s=3100,edgecolors=".25",linewidths=1.2,zorder=3)
        for x,y,value in zip(xs,ys,values):
            ax.text(x,y,f"{value:.3f}",ha="center",va="center",fontsize=20,zorder=4)
        cax=fig.add_axes([.925,.29,.012,.38])
        ticks=[1.] if max(values)-min(values)<1e-10 else [min(values),1,max(values)]
        bar=fig.colorbar(dots,cax=cax,ticks=ticks)
        bar.ax.set_yticklabels([f"{v:.4f}" for v in ticks],fontsize=17)
        bar.set_label(r"$\rho_i$",fontsize=24,labelpad=10)
        ax.set_title(slide.get("density_reference_label", "DMRG χ=128"),fontsize=31,pad=20)
    else:
        ax.scatter(xs,ys,c="white",edgecolors="black",s=350,linewidths=1.5,zorder=3)
        fig.text(.605,.16,f'Horizontal: {slide["central_bond_counts"]["x"]} x bonds',color="#c54545",fontsize=25)
        fig.text(.605,.115,f'Vertical: {slide["central_bond_counts"]["y"]} y bonds, including wrap',color="#356dc0",fontsize=25)
    ax.set(xlim=(-.55,Lx-.35),ylim=(-.5,Ly-.3),aspect="equal",xticks=range(Lx),yticks=range(Ly))
    ax.set_xlabel("periodic x" if slide.get("pbc_x", False) else "open x",fontsize=26)
    ax.set_ylabel("periodic y" if slide.get("pbc_y", True) else "open y",fontsize=26)
    ax.tick_params(labelsize=22)
    ax.spines[["top","right","bottom","left"]].set_visible(False)
    fig.text(.61,.79,slide.get("region_label", "Highlighted strip C"),fontsize=30,weight="bold")


def small_minsr(fig, slide):
    """One-step, small-sample check with exact full-sector energy measurements."""
    fig.text(.065,.80,slide["subtitle"],fontsize=31,va="top")
    ax=fig.add_axes([.085,.22,.37,.47])
    for row,color in zip(slide["cases"],["#3575a2","#c5753a"]):
        ax.plot([0,1],[slide["before"],row["after"]],marker="o",lw=3,ms=10,color=color,label=row["label"])
    ax.set_xticks([0,1],labels=["Before","After one step"])
    ax.set_ylabel("Exact variational energy / t",fontsize=28)
    ax.tick_params(labelsize=24)
    ax.ticklabel_format(axis="y",style="plain",useOffset=False)
    ax.grid(alpha=.2)
    ax.legend(frameon=False,fontsize=23)
    ax2=fig.add_axes([.53,.28,.41,.37]);ax2.axis("off")
    table_rows=[[r["label"],f'{r["after"]:.6f}',f'{r["after"]-slide["before"]:.6f}'] for r in slide["cases"]]
    tab=ax2.table(cellText=table_rows,colLabels=["Method","E after / t","ΔE / t"],cellLoc="center",bbox=[0,0,1,1],colWidths=[.32,.35,.33])
    tab.auto_set_font_size(False);tab.set_fontsize(29)
    for (row,col),cell in tab.get_celld().items():
        cell.set_edgecolor(".6")
        if row==0:
            cell.set_facecolor("#d5dcdc");cell.get_text().set_weight("bold")
    fig.text(.53,.20,slide["settings_note"],fontsize=26,va="top",linespacing=1.4)


def ed_state(fig, slide):
    """Visualize stored ED means; do not infer or sample a new state."""
    Lx, Ly = slide['Lx'], slide['Ly']
    density = slide['density']
    ax = fig.add_axes([.075, .25, .35, .48])
    xs = [i % Lx for i in range(Lx * Ly)]
    ys = [i // Lx for i in range(Lx * Ly)]
    for y in range(Ly):
        for x in range(Lx - 1):
            ax.plot([x, x+1], [y, y], color='.65', lw=2)
        ax.add_patch(FancyArrowPatch((0, y), (Lx-1, y), arrowstyle='-',
            connectionstyle='arc3,rad=.24', linestyle='--', color='.65', lw=2))
    for x in range(Lx):
        for y in range(Ly - 1):
            ax.plot([x, x], [y, y+1], color='.65', lw=2)
        ax.add_patch(FancyArrowPatch((x, 0), (x, Ly-1), arrowstyle='-',
            connectionstyle='arc3,rad=-.24', linestyle='--', color='.65', lw=2))
    ax.scatter(xs, ys, c=density, cmap='coolwarm', vmin=.95, vmax=1.05,
               s=3100, edgecolors='.3', zorder=3)
    for i, (x, y) in enumerate(zip(xs, ys)):
        ax.text(x, y, f'{density[i]:.3f}', fontsize=22, ha='center', va='center', zorder=4)
    ax.set(xlim=(-.5, Lx-.5), ylim=(-.5, Ly-.5), aspect='equal', xticks=range(Lx), yticks=range(Ly))
    ax.set_xlabel('Periodic x', fontsize=28)
    ax.set_ylabel('Periodic y', fontsize=28)
    ax.set_title(r'Site density $\rho_i=\langle n_i\rangle$', fontsize=35, pad=24)
    ax.spines[['top', 'right', 'bottom', 'left']].set_visible(False)
    fig.text(.10, .15, 'Uniform mean density; not one occupation configuration.', fontsize=25)
    matrix = fig.add_axes([.55, .24, .32, .50])
    im = matrix.imshow(slide['rho'], vmin=0, vmax=1, cmap='viridis', origin='lower')
    for i, row in enumerate(slide['rho']):
        for j, value in enumerate(row):
            matrix.text(j, i, f'{value:.3f}', ha='center', va='center', fontsize=17,
                        color='black' if value > .7 else 'white')
    matrix.set(xticks=range(Lx*Ly), yticks=range(Lx*Ly))
    matrix.set_xlabel('Physical site j', fontsize=28)
    matrix.set_ylabel('Physical site i', fontsize=28)
    matrix.set_title(r'One-body matrix $G_{ij}=\langle b_i^\dagger b_j\rangle$', fontsize=33, pad=24)
    bar = fig.colorbar(im, cax=fig.add_axes([.895, .27, .014, .42]))
    bar.ax.tick_params(labelsize=23)
    fig.text(.55, .15, f'NN means: Gx = {slide["Gx"]:.6f},  Gy = {slide["Gy"]:.6f}', fontsize=27)


def render(slide, number):
    kind = slide["kind"]
    fig = page(slide.get("title", ""), slide.get("footer", ""), number)
    if kind == "section":
        fig.clear()
        fig.text(.5, .54, slide['title'], fontsize=90, weight='bold',
                 ha='center', va='center', linespacing=1.4)
    elif kind == "title":
        fig.clear()
        fig.text(.08, .67, slide["title"], fontsize=95, weight="bold", va="top")
        fig.text(.08, .30, slide.get("subtitle", ""), fontsize=43)
        fig.text(.08, .17, slide.get("date", ""), fontsize=38)
    elif kind == "table":
        table(fig, slide)
    elif kind == "plots":
        plot_panels(fig, slide)
    elif kind == "architecture":
        architecture(fig, slide)
    elif kind == "observable":
        observable(fig, slide)
    elif kind == "small_minsr":
        small_minsr(fig, slide)
    elif kind == "ed_state":
        ed_state(fig, slide)
    elif kind in ("text", "equations", "lattice"):
        text_lines(fig, slide["lines"], fontsize=slide.get("fontsize", 44),
                   y=slide.get("y", .75), step=slide.get("step", .14))
        if kind == "lattice":
            lattice(fig, slide)
    else:
        raise ValueError(f"unknown slide kind: {kind}")
    return fig


def input_hashes(inputs):
    """Keep user-retired style templates distinct from required numerical inputs."""
    retirement = ROOT / 'slides/retired-style-references.json'
    retired = json.loads(retirement.read_text())['retired_style_references'] if retirement.exists() else {}
    allowed = {f'BoseNQS{date}-2026.pdf' for date in ('05-29', '06-05', '07-07', '07-31')}
    if set(retired) - allowed:
        raise ValueError('Only the four original style templates may be retired')
    hashes, omitted = {}, {}
    for path in inputs:
        name = str(path.relative_to(ROOT))
        if not path.exists() and name in retired:
            omitted[name] = retired[name]
        else:
            hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    if retirement.exists():
        hashes[str(retirement.relative_to(ROOT))] = hashlib.sha256(retirement.read_bytes()).hexdigest()
    return hashes, omitted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("BoseNQS2D-2026-10-31.pdf"))
    args = parser.parse_args()
    deck_path, output = resolve(args.deck), resolve(args.output)
    deck = json.loads(deck_path.read_text())
    if deck.get("status") not in ("complete", "audited_progress_snapshot"):
        raise ValueError("final PDF requires a complete, audited deck manifest")
    if deck.get("status") == "audited_progress_snapshot" and "-progress-" not in output.name:
        raise ValueError("progress snapshots require an explicit -progress- output filename")
    if re.fullmatch(r"BoseNQS\d{2}-\d{2}-\d{4}\.pdf", output.name):
        raise ValueError("refusing to replace an original 1D PDF")
    inputs = [deck_path, Path(__file__).resolve()] + [resolve(p) for p in deck["sources"]]
    hashes, retired = input_hashes(inputs)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".slides-stage-") as directory:
        staged = Path(directory) / "slides.pdf"
        with PdfPages(staged, metadata={"Title": deck["title"], "Author": "Bose-Hubbard NQS project"}) as pdf:
            for number, slide in enumerate(deck["slides"], 1):
                fig = render(slide, number)
                pdf.savefig(fig)
                plt.close(fig)
        staged.replace(output)
    provenance = {"created_at": datetime.now(timezone.utc).isoformat(), "status": deck["status"], "pages": len(deck["slides"]),
                  "page_points": [1920, 1080], "sources_sha256": hashes,
                  "pdf_sha256": hashlib.sha256(output.read_bytes()).hexdigest()}
    if retired:
        provenance['intentionally_retired_style_references'] = retired
    output.with_suffix(".provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
