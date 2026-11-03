"""Diagrams for the reviewed group-CNN presentation; no slide replacement rules.

All numerical kernel examples are illustrative, separate from trained weights.
"""
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle

BLUE, GREEN = "#286b9b", "#388469"
CUSTOM = {"irrep_matrices", "cnn_encoding", "cnn_equivariance", "feature_axes",
          "lifting_maps", "feature_action", "group_mixing", "group_pipeline"}


def arrow(fig, start, end, label=None):
    fig.add_artist(FancyArrowPatch(start, end, transform=fig.transFigure,
                   arrowstyle="-|>", mutation_scale=24, color=BLUE, linewidth=2))
    if label:
        fig.text((start[0]+end[0])/2, (start[1]+end[1])/2+.018, label,
                 ha="center", fontsize=26, color=BLUE)


def heatmap(fig, values, box, title, fs=23):
    ax = fig.add_axes(box)
    ax.imshow(values, origin="lower", cmap="Blues", vmin=0, vmax=3)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=28, pad=12)
    for y in range(3):
        for x in range(3):
            v = values[y,x]
            ax.text(x,y,f"{v:.3g}",ha="center",va="center",fontsize=fs,color="white" if v>1.5 else "black")
    return ax


def responses(X=None):
    """Exact illustrative scalar correlations, independent of trained weights."""
    if X is None:
        X = np.zeros((3,3)); X[0,1] = 1
    kernel = np.array([[0,1,0],[2,0,0],[0,0,3]])
    result = []
    for m in (0,1):
        for r in range(4):
            K = np.rot90(np.flip(kernel,axis=0) if m else kernel,-r)
            F = np.zeros((3,3))
            for y in range(3):
                for x in range(3):
                    F[y,x] = sum(K[dy+1,dx+1]*X[(y+dy)%3,(x+dx)%3]
                                 for dy in (-1,0,1) for dx in (-1,0,1))
            result.append(F)
    return X, result


def rotate_sites(values):
    """Active R about (0,0) modulo 3, rather than array-center rotation."""
    return np.array([[values[(-x)%3,y] for x in range(3)] for y in range(3)])


def draw(fig, slide):
    kind = slide["kind"]
    if kind == "irrep_matrices":
        fig.text(.07,.77,"One-dimensional example A1: a symmetric state is unchanged.",fontsize=39)
        fig.text(.10,.65,r"$D_{A_1}(R)=[1],\qquad D_{A_1}(M)=[1]$",fontsize=45)
        fig.text(.07,.49,"Two-dimensional example E: the operators mix a pair of basis states.",fontsize=39)
        # Separate rows avoid relying on unsupported matrix environments in mathtext.
        for x, label, rows in ((.14,"D_E(R)",[[0,-1],[1,0]]),(.51,"D_E(M)",[[1,0],[0,-1]])):
            fig.text(x,.32,f"${label}=$",fontsize=46)
            ax=fig.add_axes([x+.16,.25,.16,.20]);ax.axis("off")
            t=ax.table(cellText=rows,cellLoc="center",bbox=[.12,.04,.76,.92]);t.auto_set_font_size(False);t.set_fontsize(40)
            for cell in t.get_celld().values():cell.set_edgecolor("white")
            for xs in ([.08,.02,.02,.08],[.92,.98,.98,.92]):
                ax.plot(xs,[0,0,1,1],transform=ax.transAxes,color="black",lw=2,clip_on=False)
        fig.text(.08,.19,r"$U_R|v_1\rangle=|v_2\rangle,\quad U_R|v_2\rangle=-|v_1\rangle$",fontsize=34)
        fig.text(.55,.19,r"$U_M|v_1\rangle=|v_1\rangle,\quad U_M|v_2\rangle=-|v_2\rangle$",fontsize=34)
        fig.text(.07,.09,"No common one-dimensional subspace survives both matrices: R exchanges the axes fixed by M.",fontsize=33)
    elif kind == "cnn_encoding":
        heatmap(fig,np.array([[0,2,1],[1,1,1],[1,1,1]]),[.06,.39,.21,.31],"Occupation n(r)",fs=31)
        fig.text(.08,.27,r"At a site with $n=2$:",fontsize=33)
        fig.text(.08,.18,r"$(X_0,X_1,X_2,X_3,X_4)=(0,0,1,0,0)$",fontsize=30)
        arrow(fig,(.29,.55),(.40,.55),"one-hot")
        for c in range(5):
            x=.43+c*.10
            fig.add_artist(Rectangle((x,.39+c*.018),.08,.24,transform=fig.transFigure,
                           facecolor="#edf3f6",edgecolor=BLUE,linewidth=2))
            fig.text(x+.04,.51+c*.018,f"$X_{c}$",fontsize=29,ha="center")
        fig.text(.43,.28,"Five input maps: one for each allowed local occupation.",fontsize=31)
        fig.text(.43,.17,"Learned 3 × 3 kernels combine them into C output maps Fₒ(r).",fontsize=30)
        fig.text(.07,.08,"Relative coordinates δ specify the nine positions around r; all positions wrap periodically.",fontsize=32)
    elif kind == "cnn_equivariance":
        X,F=responses();F=[v/(1+np.exp(-v)) for v in F]
        Y=np.roll(X,1,axis=1);G=np.roll(F[0],1,axis=1)
        for values,box,title in [(X,[.06,.51,.16,.23],"Input X"),(Y,[.38,.51,.16,.23],"Translated input"),
                                 (F[0],[.06,.16,.16,.23],"Feature F[X]"),(G,[.38,.16,.16,.23],"Translated feature")]:
            heatmap(fig,values,box,title,fs=24)
        arrow(fig,(.235,.62),(.36,.62),"Tₓ")
        arrow(fig,(.235,.27),(.36,.27),"Tₓ")
        arrow(fig,(.14,.48),(.14,.42))
        arrow(fig,(.46,.48),(.46,.42))
        fig.text(.60,.70,"Equivariant feature map",fontsize=36,weight="bold")
        fig.text(.60,.59,r"$F[T_xX]=T_xF[X]$",fontsize=40)
        fig.text(.60,.43,"Desired invariant log amplitude",fontsize=35,weight="bold")
        fig.text(.60,.32,r"$f_\theta(T_x\mathbf{n})=f_\theta(\mathbf{n})$",fontsize=39)
        fig.text(.60,.18,"Pool over sites to remove motion.\nA dense head with different weights\nat different sites need not be invariant.",fontsize=30,linespacing=1.5)
    elif kind == "feature_axes":
        labels=["1","R","R²","R³","M","RM","R²M","R³M"]
        for h,label in enumerate(labels):
            x=.055+.112*h
            ax=heatmap(fig,np.zeros((3,3)),[x,.40,.08,.20],label,fs=0)
            for txt in ax.texts:txt.set_visible(False)
            for v in [.5,1.5]:ax.axhline(v,color="white",lw=2);ax.axvline(v,color="white",lw=2)
        fig.text(.07,.75,r"$F_p^{(1)}(\mathbf{r},h)$: a feature value indexed by channel, site and orientation.",fontsize=40)
        fig.text(.07,.28,r"$p=1,\ldots,C$: learned feature type. The row above shows one fixed p.",fontsize=36)
        fig.text(.07,.18,r"$\mathbf{r}=(x,y)$: a cell in a map. $h\in D_4$: which map in the row.",fontsize=36)
        fig.text(.07,.08,r"Ordinary CNN: $C\times3\times3$ values. Lifted CNN: $C\times8\times3\times3$ values.",fontsize=36)
    elif kind == "lifting_maps":
        X,F=responses()
        heatmap(fig,X,[.05,.36,.18,.28],r"One input map $X_2$",fs=28)
        arrow(fig,(.25,.50),(.32,.50),"lift")
        for h,values in enumerate(F):
            label=["1","R","R²","R³","M","RM","R²M","R³M"][h]
            heatmap(fig,values,[.35+(h%4)*.155,.49-(h//4)*.28,.12,.19],label,fs=20)
        fig.text(.06,.12,"Each map is the response to a different rotated/reflected copy of the same filter.",fontsize=34)
        fig.text(.06,.06,"Illustration: only one input/output channel, zero bias, identity activation; the full layer sums all c and uses SiLU.",fontsize=26)
    elif kind == "feature_action":
        X,F=responses();RX=rotate_sites(X);_,RF=responses(RX)
        assert np.array_equal(RF[1],rotate_sites(F[0]))
        for values,box,title in [(X,[.06,.51,.15,.22],"Input X"),(RX,[.38,.51,.15,.22],"Rotated input RX"),
                (F[0],[.06,.16,.15,.22],"Old map h = 1"),(RF[1],[.38,.16,.15,.22],"New map h = R")]:
            heatmap(fig,values,box,title,fs=23)
        arrow(fig,(.23,.62),(.36,.62),"R")
        arrow(fig,(.23,.27),(.36,.27),"R")
        arrow(fig,(.135,.48),(.135,.42));arrow(fig,(.455,.48),(.455,.42))
        fig.text(.58,.72,"Group-equivariant features",fontsize=36,weight="bold")
        fig.text(.58,.61,r"$F_o[g\mathbf{n}](\mathbf{r},h)$",fontsize=38)
        fig.text(.60,.52,r"$=F_o[\mathbf{n}](q^{-1}(\mathbf{r}-\mathbf{t}),q^{-1}h)$",fontsize=35)
        fig.text(.58,.37,"Both the site and the orientation label change.",fontsize=30)
        fig.text(.58,.24,"Example: after rotating the input by R,\nthe new h = R map comes from the\nold h = 1 map, spatially rotated by R.",fontsize=30,linespacing=1.45)
    elif kind == "group_mixing":
        labels=["1","R","R²","R³","M","RM","R²M","R³M"]
        for a,label in enumerate(labels):
            y=.77-.078*a
            fig.add_artist(Rectangle((.07,y-.027),.21,.055,transform=fig.transFigure,
                           facecolor="#eef3f5",edgecolor=BLUE))
            fig.text(.175,y,f"Input a = {label}",fontsize=25,ha="center",va="center")
            arrow(fig,(.29,y),(.50,.48))
        fig.text(.53,.48,r"$\sum_{p,\boldsymbol{\delta},a}$",fontsize=47,ha="center",va="center")
        arrow(fig,(.57,.48),(.69,.48),"SiLU")
        fig.add_artist(Rectangle((.71,.40),.23,.16,transform=fig.transFigure,facecolor="#edf4ef",edgecolor=GREEN,lw=2))
        fig.text(.825,.48,"Output h = R",fontsize=34,ha="center",va="center")
        fig.text(.36,.74,r"One fixed output channel $o$, at site $\mathbf{r}$.",fontsize=34)
        fig.text(.39,.24,r"Relative orientation $s=R^{-1}a$:",fontsize=35)
        fig.text(.39,.15,r"$a=R\Rightarrow s=1;\quad a=1\Rightarrow s=R^3;\quad a=M\Rightarrow s=R^3M$.",fontsize=32)
        fig.text(.07,.065,"Every arrow uses Wₒₚ(R⁻¹δ, R⁻¹a), with sums over input channels p and relative positions δ.",fontsize=30)
    elif kind == "group_pipeline":
        boxes=[("Input X","5 × 3 × 3"),("Lift + SiLU","16 × 8 × 3 × 3"),
               ("Group conv + SiLU","16 × 8 × 3 × 3"),("Mean over r, h","16 values zₒ"),("Linear head","Scalar fθ")]
        for i,(name,shape) in enumerate(boxes):
            x=.04+i*.19
            fig.add_artist(Rectangle((x,.48),.165,.20,transform=fig.transFigure,facecolor="#edf3f6",edgecolor=BLUE,lw=2))
            fig.text(x+.0825,.61,name,ha="center",fontsize=28)
            fig.text(x+.0825,.53,shape,ha="center",fontsize=25)
            if i<4:arrow(fig,(x+.167,.58),(x+.187,.58))
        fig.text(.07,.34,r"$z_o=\frac{1}{72}\sum_{\mathbf{r},h}F_o^{(2)}(\mathbf{r},h),\qquad f_\theta(\mathbf{n})=\sum_o w_oz_o+\eta$",fontsize=43)
        fig.text(.07,.21,"A symmetry operation only permutes the 72 positions (r, h): the mean is unchanged.",fontsize=35)
        fig.text(.07,.09,r"$f_\theta(g\mathbf{n})=f_\theta(\mathbf{n})\quad\Longrightarrow\quad\psi_\theta(g\mathbf{n})=\psi_\theta(\mathbf{n}),\qquad\psi_\theta=e^{f_\theta}/\sqrt{Z_\theta}$",fontsize=40)
    else:
        raise ValueError(kind)
