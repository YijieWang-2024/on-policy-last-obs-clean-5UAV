from argparse import Namespace
from pathlib import Path
import json
import sys
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic import R_Actor
from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic_attention import R_Actor_Attention
from onpolicy.envs.mec.env_maker import WrappedMECEnv
from onpolicy.envs.mec.mec import _cartesian_flight_velocity, _offload_delay
from onpolicy.envs.mec.vec_normalize import Normer
from onpolicy.scripts.eval.render_dynamic_mappo_episode import frozen_normalize

ROOT = Path(__file__).resolve().parent
RUNS = {name: ROOT / f"{name}_seed100000" for name in ("q0", "q1", "q2")}
SEED = 100000

def corr(x, y):
    return float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 and np.std(x) and np.std(y) else 0.0

def evaluate(name, folder, resource_mode="learned"):
    ckpt = folder / "checkpoint_snapshot"
    args = Namespace(**json.loads((ckpt / "args.json").read_text()))
    args.n_rollout_threads = args.n_training_threads = 1
    args.model_dir = str(ckpt)
    np.random.seed(SEED); torch.manual_seed(SEED); torch.set_num_threads(1)
    env = WrappedMECEnv(args=args); env.seed(SEED)
    actor_cls = R_Actor_Attention if args.use_atten_actor else R_Actor
    actors=[]; normers=[]
    for i in range(args.n_UAVs):
        actor=actor_cls(args,env.observation_space[i],env.action_space[i],torch.device("cpu"))
        actor.load_state_dict(torch.load(ckpt/f"actor_agent{i}.pt",map_location="cpu",weights_only=True));actor.eval();actors.append(actor)
        normer=Normer(args=args,obs_space=env.observation_space[i],states_space=env.share_observation_space[i]);normer.load(ckpt/f"normer{i}.pkl");normers.append(normer)
    obs,states,avail,_,attention=env.reset();obs,states=frozen_normalize(normers,obs,states)
    rnn=np.zeros((args.n_UAVs,args.recurrent_N,args.hidden_size),np.float32);masks=np.ones((args.n_UAVs,1),np.float32)
    totals={"candidate":0,"selected":0,"equal_candidate_feasible":0,"equal_selected_feasible":0}
    flight=[]; bw=[]; cpu=[]; data=[]; cycles=[]; deadlines=[]; bw_cv=[];cpu_cv=[]
    final_info={}
    with torch.no_grad():
        for _ in range(args.episode_length):
            raw=[]
            for i,a in enumerate(actors):
                kwargs={"attention_active_mask":attention[i:i+1]} if args.use_atten_actor else {}
                action,_,nr=a(obs[i:i+1],rnn[i:i+1],masks[i:i+1],avail[i:i+1],deterministic=True,**kwargs)
                raw.append(action.numpy()[0]);rnn[i]=nr.numpy()[0]
            raw=np.asarray(raw)
            k=args.max_GUs_in_range
            if resource_mode == "equal":
                raw[:,2+k:2+3*k]=1.0
            elif resource_mode == "equal_selected":
                # Preserve the exact UAV-user pairs accepted by the learned
                # resource outputs, then allocate equally only within that set.
                learned_transformed=env.env.transform_uav_actions(raw)
                learned_processed=env.env.process_actions(learned_transformed)
                raw[:,2:2+k]=0.0
                for u in range(args.n_UAVs):
                    for local_pos, gu in enumerate(env.nearby_gus_of_uavs[u,:k]):
                        if gu >= 0 and learned_processed[u,2+int(gu)] > 0:
                            raw[u,2+local_pos]=1.0
                raw[:,2+k:2+3*k]=1.0
            elif resource_mode != "learned":
                raise ValueError(resource_mode)
            transformed=env.env.transform_uav_actions(raw)
            processed=env.env.process_actions(transformed)
            n=args.n_GUs
            candidate=transformed[:,2:2+n]>=.5
            selected=processed[:,2:2+n]>0
            candidate_users=np.flatnonzero(np.any(candidate,axis=0)&env.active_md_mask)
            selected_users=np.flatnonzero(np.any(selected,axis=0)&env.active_md_mask)
            totals["candidate"]+=len(candidate_users);totals["selected"]+=len(selected_users)

            # Resolve duplicate candidate associations to the nearest proposing UAV.
            cand_uavs=[];cand_gus=[]
            for gu in candidate_users:
                ids=np.flatnonzero(candidate[:,gu]);u=int(ids[np.argmin(env.uav_gu_distances_3d[ids,gu])]);cand_uavs.append(u);cand_gus.append(int(gu))
            if cand_gus:
                cb=np.zeros((args.n_UAVs,n));cc=np.zeros((args.n_UAVs,n))
                for u in range(args.n_UAVs):
                    ids=[g for uu,g in zip(cand_uavs,cand_gus) if uu==u]
                    if ids:cb[u,ids]=1/len(ids);cc[u,ids]=1/len(ids)
                td,ed=_offload_delay(env.gu_tasks,np.asarray(cand_uavs),np.asarray(cand_gus),cb*args.B,cc*args.F_m,env.channel_gains)
                totals["equal_candidate_feasible"]+=int(np.sum(td+ed<=env.gu_tasks[cand_gus,2]))

            pb=processed[:,2+n:2+2*n];pc=processed[:,2+2*n:]
            if selected_users.size:
                su=np.argmax(selected[:,selected_users],axis=0)
                eb=np.zeros_like(pb);ec=np.zeros_like(pc)
                for u in range(args.n_UAVs):
                    ids=selected_users[su==u]
                    if len(ids):eb[u,ids]=1/len(ids);ec[u,ids]=1/len(ids)
                td,ed=_offload_delay(env.gu_tasks,su,selected_users,eb*args.B,ec*args.F_m,env.channel_gains)
                totals["equal_selected_feasible"]+=int(np.sum(td+ed<=env.gu_tasks[selected_users,2]))
                bw.extend(pb[su,selected_users]);cpu.extend(pc[su,selected_users]);data.extend(env.gu_tasks[selected_users,0]);cycles.extend(env.gu_tasks[selected_users,1]);deadlines.extend(env.gu_tasks[selected_users,2])
            for u in range(args.n_UAVs):
                vals=pb[u,pb[u]>0];bw_cv.append(float(np.std(vals)/np.mean(vals)) if len(vals)>1 else 0.)
                vals=pc[u,pc[u]>0];cpu_cv.append(float(np.std(vals)/np.mean(vals)) if len(vals)>1 else 0.)
            flight.extend(np.linalg.norm(_cartesian_flight_velocity(raw[:,:2],args.v_max),axis=1) if args.cartesian_flight else raw[:,1]*args.v_max)
            obs,states,_,dones,final_info,avail,_,attention=env.step(raw);obs,states=frozen_normalize(normers,obs,states);masks[:]=0 if np.all(dones) else 1
    env.close();bw=np.asarray(bw);cpu=np.asarray(cpu)
    result={
      "resource_mode":resource_mode,
      "true60":float(final_info[0]["system_performance_true_all_GUs"]) if isinstance(final_info,list) else float(np.asarray(final_info["system_performance_true_all_GUs"])[0]),
      "completion_ratio":float(final_info[0]["complete_task_ratio"]) if isinstance(final_info,list) else float(final_info["complete_task_ratio"]),
      "candidate_offloads":totals["candidate"],"selected_offloads":totals["selected"],
      "postprocess_cancel_ratio":1-totals["selected"]/max(totals["candidate"],1),
      "equal_resource_candidate_feasible_ratio":totals["equal_candidate_feasible"]/max(totals["candidate"],1),
      "equal_resource_on_selected_feasible_ratio":totals["equal_selected_feasible"]/max(totals["selected"],1),
      "commanded_speed_mean_mps":float(np.mean(flight)),"commanded_speed_p95_mps":float(np.percentile(flight,95)),
      "bandwidth_fraction_cv_mean":float(np.mean(bw_cv)),"cpu_fraction_cv_mean":float(np.mean(cpu_cv)),
      "bandwidth_corr_input_data":corr(bw,np.asarray(data)),"cpu_corr_required_cycles":corr(cpu,np.asarray(cycles)),
      "bandwidth_corr_deadline":corr(bw,np.asarray(deadlines)),"cpu_corr_deadline":corr(cpu,np.asarray(deadlines)),
    }
    return result

if __name__ == "__main__":
    results={
        name:{mode:evaluate(name,folder,mode) for mode in ("learned","equal","equal_selected")}
        for name,folder in RUNS.items()
    }
    assert all(v["learned"]["selected_offloads"]>0 for v in results.values())
    (ROOT/"raw_action_audit.json").write_text(json.dumps(results,indent=2),encoding="utf-8")
    print(json.dumps(results,indent=2))
