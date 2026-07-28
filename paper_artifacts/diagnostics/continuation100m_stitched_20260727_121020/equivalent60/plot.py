from pathlib import Path
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing import event_accumulator

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
RESULTS = ROOT / "onpolicy" / "scripts" / "results" / "mec" / "mappo"
FIG4 = ROOT / "paper_artifacts" / "published_figures" / "fig4.png"
TAG = "agent0/system_performance_equivalent_full_GUs"
DEN = 64 * 400
OFFSET = 50_000_000
X0, X3500, Y500K, Y400K = 390.0, 2811.0, 397.0, 784.0
PX_PER_EP = (X3500 - X0) / 3500.0
X4000 = X0 + 4000 * PX_PER_EP

def smooth3(y):
    return np.array([y[max(0, i-1):min(len(y), i+2)].mean() for i in range(len(y))])

def load(log_dir):
    event_file = sorted(log_dir.glob("events.out.tfevents.*"))[-1]
    acc = event_accumulator.EventAccumulator(str(event_file), size_guidance={event_accumulator.SCALARS: 0})
    acc.Reload()
    events = acc.Scalars(TAG)
    return np.array([e.step for e in events], dtype=np.int64), np.array([e.value for e in events], float)

def stitched(prefix):
    first = RESULTS / f"regional_dynamic_strict1p5_init110_220_330_440_{prefix}_gae095_seed2_50m" / "run1" / "logs"
    second = RESULTS / f"regional_dynamic_strict1p5_init110_220_330_440_{prefix}_gae095_seed2_warm50to100m" / "run1" / "logs"
    s1, y1 = load(first); s2, y2 = load(second)
    steps = np.concatenate([s1, s2 + OFFSET])
    values = np.concatenate([y1, y2])
    segments = np.array(["0-50M"] * len(y1) + ["50-100M"] * len(y2))
    order = np.argsort(steps, kind="stable")
    steps, values, segments = steps[order], values[order], segments[order]
    keep = np.r_[True, np.diff(steps) > 0]
    return steps[keep], values[keep], segments[keep]

def live(name):
    steps, values = load(RESULTS / name / "run1" / "logs")
    return steps, values, np.array(["live"] * len(values))

def pixels(x, y):
    return X0 + x * PX_PER_EP, Y500K + (500000-y) * (Y400K-Y500K)/100000

def main():
    specs = [("MAPPO strict 1+5 (stitched 100M)", "mappo", "#17becf", "-", True),
             ("DC-PPO strict 1+5 (stitched 100M)", "dcppo", "#8c564b", "--", True),
             ("E1 spatial-flight (live)", "regional_dynamic_strict1p5_cartesian_spatialflight_e1_seed2_100m_r64_20260727", "#e377c2", "-", False),
             ("E2 + reset curriculum (live)", "regional_dynamic_strict1p5_cartesian_spatialflight_curriculum_e2_seed2_100m_r64_20260727", "#111111", "--", False)]
    curves=[]
    for label, source, color, style, is_stitched in specs:
        steps, raw, segments = stitched(source) if is_stitched else live(source)
        curves.append(dict(label=label, steps=steps, x=steps/DEN, raw=raw,
                           smooth=smooth3(raw), segments=segments, color=color, style=style))
    with (OUT/"plotted_curves.csv").open("w", newline="", encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["series","segment","cumulative_steps","learning_episode","raw_value","three_point_smoothed_value"])
        for c in curves:
            for row in zip(c["segments"],c["steps"],c["x"],c["raw"],c["smooth"]): w.writerow([c["label"],*row])
    with (OUT/"curve_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["series","points","last_steps","last_episode","last20","last50","best20_full","first50m_last20","second50m_first20","second50m_last20"])
        for c in curves:
            cut=np.searchsorted(c["steps"], OFFSET, side="right")
            roll=np.convolve(c["raw"],np.ones(20)/20,mode="valid")
            second=c["raw"][cut:]
            w.writerow([c["label"],len(c["raw"]),c["steps"][-1],c["x"][-1],c["raw"][-20:].mean(),c["raw"][-50:].mean(),roll.max(),c["raw"][:cut][-20:].mean(),second[:20].mean() if len(second) else "",second[-20:].mean() if len(second) else ""])

    bg=mpimg.imread(FIG4); h,w=bg.shape[:2]
    new_w=int(np.ceil(X4000+150)); canvas=np.ones((h,new_w,bg.shape[2]),dtype=bg.dtype); canvas[:,:w]=bg
    fig=plt.figure(figsize=(new_w/300,h/300),dpi=300); ax=fig.add_axes([0,0,1,1]); ax.imshow(canvas,origin="upper")
    # Extend the horizontal plot guides into the added 3500-4000 region.
    for yval in [200000,300000,400000,500000]:
        _,py=pixels(0,yval); ax.plot([X3500,X4000],[py,py],color="#d0d0d0",linestyle="--",linewidth=.8,zorder=1)
    ax.plot([X4000,X4000],[32,1778],color="#d0d0d0",linestyle="--",linewidth=.8,zorder=1)
    boundary_ep=OFFSET/DEN; bx,_=pixels(boundary_ep,500000)
    ax.plot([bx,bx],[32,1778],color="#555555",linestyle=":",linewidth=1.1,zorder=4)
    ax.text(bx+8,80,"50M continuation",fontsize=8,color="#444444",rotation=90,va="top")
    for c in curves:
        px,py=pixels(c["x"],c["smooth"]); v=(px>=X0)&(px<=X4000)&(py>=32)&(py<=1778)
        ax.plot(px[v],py[v],color=c["color"],linestyle=c["style"],linewidth=2.8,label=c["label"],zorder=5)
    # Add the 4000 tick in the extended area.
    ax.text(X4000,1838,"4000",ha="center",va="top",fontsize=10,color="#2b2b2b")
    ax.plot([X4000,X4000],[1778,1790],color="#333333",linewidth=1)
    ax.set_xlim(-.5,new_w-.5); ax.set_ylim(h-.5,-.5); ax.axis("off")
    ax.legend(loc="lower right",bbox_to_anchor=(.975,.075),frameon=True,framealpha=.94,fontsize=7.5,title="Strict 1+5 comparison",title_fontsize=7.8,handlelength=3)
    fig.savefig(OUT/"plot.png",dpi=300,bbox_inches="tight",pad_inches=0.02); plt.close(fig)

if __name__ == "__main__": main()
