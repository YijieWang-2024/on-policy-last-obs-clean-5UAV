from pathlib import Path
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from tensorboard.backend.event_processing import event_accumulator

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[3]
RESULTS=ROOT/"onpolicy"/"scripts"/"results"/"mec"/"mappo"
OLD=RESULTS/"check"; DEN=64*400; N=5; METRIC="system_performance_individual"

def smooth3(y): return np.array([y[max(0,i-1):min(len(y),i+2)].mean() for i in range(len(y))])
def read(file,tag):
    a=event_accumulator.EventAccumulator(str(file),size_guidance={event_accumulator.SCALARS:0});a.Reload();e=a.Scalars(tag)
    return np.array([z.step/DEN for z in e],float),np.array([z.value for z in e],float)
def old_sum(run):
    data=[]
    for i in range(N):
        p=OLD/run/"logs"/f"agent{i}"/METRIC/f"agent{i}"/METRIC;data.append(read(sorted(p.glob("events.out.tfevents.*"))[-1],f"agent{i}/{METRIC}"))
    n=min(len(y) for x,y in data);return data[0][0][:n],np.sum([y[:n] for x,y in data],axis=0)
def current_sum(run):
    file=sorted((run/"logs").glob("events.out.tfevents.*"))[-1];data=[read(file,f"agent{i}/{METRIC}") for i in range(N)];n=min(len(y) for x,y in data)
    return data[0][0][:n],np.sum([y[:n] for x,y in data],axis=0)
def historical(runs):
    curves=[]
    for run in runs:
        x,y=old_sum(run);m=x<=3499;curves.append((x[m],smooth3(y[m])))
    start=max(x[0] for x,y in curves);end=min(x[-1] for x,y in curves);count=min(len(x) for x,y in curves);cx=np.linspace(start,end,count);a=np.vstack([np.interp(cx,x,y) for x,y in curves])
    return cx,a.mean(0),a.std(0)
def fmt(v,_):return f"{v/1000:g}k" if abs(v)>=1000 else f"{v:g}"

def main():
    hm_x,hm,hs_m=historical(["run313","run321","run322"]);hd_x,hd,hs_d=historical(["run301","run304","run30901"])
    mdir=RESULTS/"regional_dynamic_strict1p5_init110_220_330_440_mappo_gae095_seed2_warm50to100m"/"run1";ddir=RESULTS/"regional_dynamic_strict1p5_init110_220_330_440_dcppo_gae095_seed2_warm50to100m"/"run1"
    mx,mraw=current_sum(mdir);dx,draw=current_sum(ddir)
    series=[
      dict(label="MAPPO (fixed 60, 3 seeds)",kind="historical",x=hm_x,mean=hm,std=hs_m,color="#1f77b4",style="-"),
      dict(label="DC-PPO (fixed 60, 3 seeds)",kind="historical",x=hd_x,mean=hd,std=hs_d,color="#ff7f0e",style="--"),
      dict(label="MAPPO continuation (local x)",kind="current",x=mx,mean=smooth3(mraw),raw=mraw,std=None,color="#17becf",style="-"),
      dict(label="DC-PPO continuation (local x)",kind="current",x=dx,mean=smooth3(draw),raw=draw,std=None,color="#8c564b",style="--")]
    with (OUT/"plotted_curves.csv").open("w",newline="",encoding="utf-8") as f:
      w=csv.writer(f);w.writerow(["series","source_kind","local_learning_episode","mean","std"])
      for c in series:
       for i,(x,y) in enumerate(zip(c["x"],c["mean"])):w.writerow([c["label"],c["kind"],float(x),float(y),"" if c["std"] is None else float(c["std"][i])])
    with (OUT/"curve_summary.csv").open("w",newline="",encoding="utf-8") as f:
      w=csv.writer(f);w.writerow(["series","endpoint_local_episode","value_at_own_endpoint","last20_mean_if_current","last50_mean_if_current","comparison_value_at_current_endpoint"])
      for c in series:
       if c["kind"]=="current":w.writerow([c["label"],c["x"][-1],c["mean"][-1],c["raw"][-20:].mean(),c["raw"][-50:].mean(),""])
      endpoint=min(mx[-1],dx[-1])
      for c in series[:2]:w.writerow([c["label"],c["x"][-1],c["mean"][-1],"","",np.interp(endpoint,c["x"],c["mean"])])
    plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["DejaVu Sans","Arial","Helvetica"],"font.size":10,"axes.labelsize":10,"xtick.labelsize":9,"ytick.labelsize":9,"legend.fontsize":8.5})
    fig,ax=plt.subplots(figsize=(7,4.5))
    for c in series:
      cur=c["kind"]=="current";ax.plot(c["x"],c["mean"],color=c["color"],linestyle=c["style"],linewidth=2.6 if cur else 2,label=c["label"],zorder=4 if cur else 2)
      if c["std"] is not None:ax.fill_between(c["x"],c["mean"]-c["std"],c["mean"]+c["std"],color=c["color"],alpha=.14,linewidth=0,zorder=1)
    ax.set_xlim(0,3500);ax.set_xlabel("Learning episodes");ax.set_ylabel("Covered-MD system performance");ax.yaxis.set_major_formatter(FuncFormatter(fmt));ax.grid(axis="both",linestyle=":",linewidth=.6,alpha=.65);ax.spines[["top","right"]].set_visible(False);ax.tick_params(direction="out",length=4);ax.legend(loc="lower right",frameon=True,framealpha=.92);fig.tight_layout();fig.savefig(OUT/"plot.png",dpi=300,bbox_inches="tight",pad_inches=.05);plt.close(fig)

if __name__=="__main__":main()
