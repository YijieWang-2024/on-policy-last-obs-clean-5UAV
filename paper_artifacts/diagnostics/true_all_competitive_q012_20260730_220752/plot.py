from pathlib import Path
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
from tensorboard.backend.event_processing import event_accumulator

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
RESULTS=ROOT/"onpolicy"/"scripts"/"results"/"mec"/"mappo"
TAG="agent0/system_performance_true_all_GUs"
OFFSET=50_000_000

def load(exp,run="run1"):
    files=list((RESULTS/exp/run/"logs").glob("events.out.tfevents.*"))
    assert files, f"No event file: {exp}/{run}"
    f=max(files,key=lambda p:p.stat().st_mtime)
    ea=event_accumulator.EventAccumulator(str(f),size_guidance={event_accumulator.SCALARS:0});ea.Reload()
    ev=ea.Scalars(TAG)
    return np.array([e.step for e in ev],dtype=np.int64),np.array([e.value for e in ev],float)

def stitched(prefix):
    s1,y1=load(f"regional_dynamic_strict1p5_init110_220_330_440_{prefix}_gae095_seed2_50m")
    s2,y2=load(f"regional_dynamic_strict1p5_init110_220_330_440_{prefix}_gae095_seed2_warm50to100m")
    return np.r_[s1,s2+OFFSET],np.r_[y1,y2]

def smooth3(y):
    return np.array([y[max(0,i-1):min(len(y),i+2)].mean() for i in range(len(y))])

def main():
    specs=[
      ("Q2 Cartesian spatial + curriculum","q2_cartesian_spatial_curriculum_completionpriority_seed2_100m_20260729","run2","#d62728","-",False),
      ("Q0 Polar standard + curriculum","q0_polar_standard_curriculum_completionpriority_seed2_100m_20260729","run1","#1f77b4","-",False),
      ("Q1 Polar spatial + curriculum","q1_polar_spatial_curriculum_completionpriority_seed2_100m_20260729","run1","#2ca02c","-",False),
      ("MAPPO strict 1+5 reference","mappo",None,"#17becf","--",True),
      ("DC-PPO strict 1+5 reference","dcppo",None,"#8c564b","--",True),
    ]
    curves=[]
    for label,source,run,color,style,is_stitched in specs:
        steps,raw=stitched(source) if is_stitched else load(source,run)
        curves.append(dict(label=label,steps=steps,raw=raw,smooth=smooth3(raw),color=color,style=style))

    with (OUT/"plotted_curves.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f);w.writerow(["series","environment_steps","raw_value","three_point_smoothed_value"])
        for c in curves:
            for row in zip(c["steps"],c["raw"],c["smooth"]):w.writerow([c["label"],*row])
    with (OUT/"curve_summary.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f);w.writerow(["series","points","last_steps","last_raw","last20_mean","last50_mean","best20_mean"])
        for c in curves:
            roll=np.convolve(c["raw"],np.ones(20)/20,mode="valid")
            w.writerow([c["label"],len(c["raw"]),c["steps"][-1],c["raw"][-1],c["raw"][-20:].mean(),c["raw"][-50:].mean(),roll.max()])

    plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["DejaVu Sans","Arial","Helvetica"],"font.size":10,"axes.labelsize":10,"xtick.labelsize":9,"ytick.labelsize":9,"legend.fontsize":8})
    fig,ax=plt.subplots(figsize=(7,4.5))
    for c in curves:ax.plot(c["steps"],c["smooth"],label=c["label"],color=c["color"],ls=c["style"],lw=2.3 if c["style"]=="-" else 1.9)
    ax.set_xlim(0,100_000_000);ax.set_xlabel("Environment steps");ax.set_ylabel("True covered-MD system performance")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v,_:f"{v/1e6:g}M"));ax.yaxis.set_major_formatter(FuncFormatter(lambda v,_:f"{v/1000:.0f}k"))
    ax.grid(axis="both",ls=":",lw=.6,alpha=.65);ax.spines[["top","right"]].set_visible(False);ax.tick_params(direction="out",length=4)
    ax.legend(loc="lower right",ncol=2,frameon=True,framealpha=.94,labelspacing=.45,handletextpad=.6,borderpad=.5)
    fig.tight_layout();fig.savefig(OUT/"plot.png",dpi=300,bbox_inches="tight",pad_inches=.05);plt.close(fig)

if __name__=="__main__":main()
