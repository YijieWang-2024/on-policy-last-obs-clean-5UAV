from pathlib import Path
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
from tensorboard.backend.event_processing import event_accumulator

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[3]
RESULTS=ROOT/"onpolicy"/"scripts"/"results"/"mec"/"mappo"
OLD=RESULTS/"check"
DEN=64*400
OFFSET=50_000_000
TAGS=[f"agent{i}/system_performance_individual" for i in range(5)]

def smooth3(y): return np.array([y[max(0,i-1):min(len(y),i+2)].mean() for i in range(len(y))])

def load(log_dir):
    f=sorted(log_dir.glob("events.out.tfevents.*"))[-1]
    ea=event_accumulator.EventAccumulator(str(f),size_guidance={event_accumulator.SCALARS:0});ea.Reload()
    all_events=[ea.Scalars(t) for t in TAGS]
    common=sorted(set.intersection(*[{e.step for e in ev} for ev in all_events]))
    maps=[{e.step:e.value for e in ev} for ev in all_events]
    return np.array(common,dtype=np.int64),np.array([sum(m[s] for m in maps) for s in common],float)

def stitched(prefix):
    stem=f"regional_dynamic_strict1p5_init110_220_330_440_{prefix}_gae095_seed2"
    s1,y1=load(RESULTS/(stem+"_50m")/"run1"/"logs")
    s2,y2=load(RESULTS/(stem+"_warm50to100m")/"run1"/"logs")
    return np.r_[s1,s2+OFFSET],np.r_[y1,y2],np.array(["0-50M"]*len(y1)+["50-100M"]*len(y2))

def live(name):
    steps,raw=load(RESULTS/name/"run1"/"logs")
    return steps,raw,np.array(["live"]*len(raw))

def historical(run_ids):
    curves=[]
    for rid in run_ids:
        per_agent=[]
        for i in range(5):
            p=OLD/f"run{rid}"/"logs"/f"agent{i}"/"system_performance_individual"/f"agent{i}"/"system_performance_individual"
            f=sorted(p.glob("events.out.tfevents.*"))[-1]
            ea=event_accumulator.EventAccumulator(str(f),size_guidance={event_accumulator.SCALARS:0});ea.Reload()
            ev=ea.Scalars(f"agent{i}/system_performance_individual")
            per_agent.append((np.array([e.step for e in ev],dtype=np.int64),np.array([e.value for e in ev],float)))
        n=min(len(y) for _,y in per_agent)
        x=per_agent[0][0][:n]/DEN; y=np.sum([v[:n] for _,v in per_agent],axis=0); mask=x<=3499
        curves.append((x[mask],y[mask]))
    start=max(x[0] for x,_ in curves);end=min(x[-1] for x,_ in curves);count=min(len(x) for x,_ in curves)
    x=np.linspace(start,end,count);ys=np.vstack([np.interp(x,cx,smooth3(cy)) for cx,cy in curves])
    return x,ys.mean(0),ys.std(0)

def main():
    # These are the three historical fixed-60 runs used in the previous comparison.
    old_specs=[("MAPPO (fixed 60, 3 seeds)",[313,321,322],"#1f77b4","-"),
               ("DC-PPO (fixed 60, 3 seeds)",[301,304,30901],"#ff7f0e","--")]
    new_specs=[("MAPPO strict 1+5 (stitched 100M)","mappo","#17becf","-",True),
               ("DC-PPO strict 1+5 (stitched 100M)","dcppo","#8c564b","--",True),
               ("E1 spatial-flight (live)","regional_dynamic_strict1p5_cartesian_spatialflight_e1_seed2_100m_r64_20260727","#e377c2","-",False),
               ("E2 + reset curriculum (live)","regional_dynamic_strict1p5_cartesian_spatialflight_curriculum_e2_seed2_100m_r64_20260727","#111111","--",False)]
    olds=[]
    for label,runs,color,style in old_specs:
        x,mean,std=historical(runs); olds.append(dict(label=label,x=x,mean=smooth3(mean),std=std,color=color,style=style))
    news=[]
    for label,source,color,style,is_stitched in new_specs:
        steps,raw,segments=stitched(source) if is_stitched else live(source); news.append(dict(label=label,steps=steps,x=steps/DEN,raw=raw,smooth=smooth3(raw),segments=segments,color=color,style=style))
    with (OUT/"plotted_curves.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f);w.writerow(["series","segment","cumulative_steps","learning_episode","raw_or_mean","three_point_smoothed"])
        for c in olds:
            for x,y in zip(c["x"],c["mean"]):w.writerow([c["label"],"historical",x*DEN,x,y,y])
        for c in news:
            for row in zip(c["segments"],c["steps"],c["x"],c["raw"],c["smooth"]):w.writerow([c["label"],*row])
    with (OUT/"curve_summary.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f);w.writerow(["series","last_steps","last_episode","last20","last50","best20_full","first50m_last20","second50m_first20","second50m_last20"])
        for c in news:
            cut=np.searchsorted(c["steps"],OFFSET,side="right");roll=np.convolve(c["raw"],np.ones(20)/20,mode="valid")
            second=c["raw"][cut:]
            w.writerow([c["label"],c["steps"][-1],c["x"][-1],c["raw"][-20:].mean(),c["raw"][-50:].mean(),roll.max(),c["raw"][:cut][-20:].mean(),second[:20].mean() if len(second) else "",second[-20:].mean() if len(second) else ""])
    plt.rcParams.update({"font.family":"sans-serif","font.size":10,"axes.labelsize":10,"xtick.labelsize":9,"ytick.labelsize":9,"legend.fontsize":8.2})
    fig,ax=plt.subplots(figsize=(7,4.5))
    for c in olds:
        ax.plot(c["x"],c["mean"],color=c["color"],ls=c["style"],lw=2,label=c["label"])
        ax.fill_between(c["x"],c["mean"]-c["std"],c["mean"]+c["std"],color=c["color"],alpha=.13,linewidth=0)
    for c in news:ax.plot(c["x"],c["smooth"],color=c["color"],ls=c["style"],lw=2.4,label=c["label"])
    boundary=OFFSET/DEN;ax.axvline(boundary,color="#555555",ls=":",lw=1.1)
    ax.text(boundary+35,ax.get_ylim()[1]*.985,"50M continuation",rotation=90,va="top",ha="left",fontsize=8,color="#444444")
    ax.set_xlim(0,4000);ax.set_xlabel("Learning episodes");ax.set_ylabel("Covered-MD system performance")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v,_:f"{v/1000:.0f}k"))
    ax.grid(axis="both",ls=":",lw=.6,alpha=.65);ax.spines[["top","right"]].set_visible(False);ax.tick_params(direction="out",length=4)
    ax.legend(loc="lower right",frameon=True,framealpha=.94,ncol=2,fontsize=7.6);fig.tight_layout();fig.savefig(OUT/"plot.png",dpi=300,bbox_inches="tight",pad_inches=.05);plt.close(fig)

if __name__=="__main__":main()
