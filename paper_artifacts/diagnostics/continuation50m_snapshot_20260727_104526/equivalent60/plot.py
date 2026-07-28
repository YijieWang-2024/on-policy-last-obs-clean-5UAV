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
PLOT_DATA = ROOT / "onpolicy" / "scripts" / "train" / "plot_data"
TAG = "agent0/system_performance_equivalent_full_GUs"
DEN = 64 * 400
X0, X3500, Y500K, Y400K = 390.0, 2811.0, 397.0, 784.0
LEFT, RIGHT, TOP, BOTTOM = 269.0, 2932.0, 32.0, 1778.0

def smooth3(y):
    return np.array([y[max(0,i-1):min(len(y),i+2)].mean() for i in range(len(y))])

def load(log_dir):
    event_file = sorted(log_dir.glob("events.out.tfevents.*"))[-1]
    acc = event_accumulator.EventAccumulator(str(event_file), size_guidance={event_accumulator.SCALARS: 0})
    acc.Reload(); events = acc.Scalars(TAG)
    return np.array([e.step / DEN for e in events], float), np.array([e.value for e in events], float)

def pixels(x, y):
    return X0 + x * (X3500-X0)/3500, Y500K + (500000-y) * (Y400K-Y500K)/100000

def main():
    specs = [
        ("MAPPO continuation (local x)", RESULTS/"regional_dynamic_strict1p5_init110_220_330_440_mappo_gae095_seed2_warm50to100m"/"run1"/"logs", "#17becf", "-"),
        ("DC-PPO continuation (local x)", RESULTS/"regional_dynamic_strict1p5_init110_220_330_440_dcppo_gae095_seed2_warm50to100m"/"run1"/"logs", "#8c564b", "--"),
    ]
    curves=[]
    for label, log_dir, color, style in specs:
        x, raw=load(log_dir); curves.append(dict(label=label,x=x,raw=raw,smooth=smooth3(raw),color=color,style=style))
    with (OUT/"plotted_curves.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["series","local_learning_episode","raw_value","three_point_smoothed_value"])
        for c in curves:
            for row in zip(c["x"],c["raw"],c["smooth"]): w.writerow([c["label"],*map(float,row)])
    historical={"MAPPO (Fig. 4)":np.load(PLOT_DATA/"mappo.npy").mean(0),"DC-PPO (Fig. 4)":np.load(PLOT_DATA/"dcppo.npy").mean(0),"ARA (Fig. 4)":np.load(PLOT_DATA/"ara_extended.npy").mean(0)}
    hx=np.arange(1,2*len(next(iter(historical.values()))),2,dtype=float)
    with (OUT/"curve_summary.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f);w.writerow(["series","points","last_local_episode","last_raw","last20_mean","last50_mean","best20_mean","historical_series","historical_value_at_same_local_episode"])
        for c in curves:
            roll=np.convolve(c["raw"],np.ones(20)/20,mode="valid")
            for old,vals in historical.items():w.writerow([c["label"],len(c["raw"]),c["x"][-1],c["raw"][-1],c["raw"][-20:].mean(),c["raw"][-50:].mean(),roll.max(),old,np.interp(c["x"][-1],hx,vals)])
    bg=mpimg.imread(FIG4); h,w=bg.shape[:2]; fig=plt.figure(figsize=(w/300,h/300),dpi=300); ax=fig.add_axes([0,0,1,1]);ax.imshow(bg,origin="upper")
    for c in curves:
        px,py=pixels(c["x"],c["smooth"]);v=(px>=LEFT)&(px<=RIGHT)&(py>=TOP)&(py<=BOTTOM);ax.plot(px[v],py[v],color=c["color"],linestyle=c["style"],linewidth=2.8,label=c["label"],zorder=5)
    ax.set_xlim(-.5,w-.5);ax.set_ylim(h-.5,-.5);ax.axis("off");ax.legend(loc="lower right",bbox_to_anchor=(.965,.075),frameon=True,framealpha=.93,fontsize=8.5,title="Continuation snapshot: equivalent 60-MD",title_fontsize=8.5,handlelength=3)
    fig.savefig(OUT/"plot.png",dpi=300,bbox_inches=None,pad_inches=0);plt.close(fig)

if __name__ == "__main__": main()
