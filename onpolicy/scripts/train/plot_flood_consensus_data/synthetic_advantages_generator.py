# 基于现有采样的数据。 生成随机的优势函数值。画最后一个图用到的随机优势
# Auto-generated synthetic advantage generator
import numpy as np, math

def gen_synthetic_advantages(N_agents, T, E, nu, mu_t, std_t, corr_c, phi_common, phi_idio, seed=0):
    rng = np.random.default_rng(seed)
    c = max(0.0, min(0.95, float(corr_c)))
    phi_c = max(-0.99, min(0.99, float(phi_common)))
    phi_e = max(-0.99, min(0.99, float(phi_idio)))

    def t_innov(shape):
        return rng.standard_t(df=float(nu), size=shape) / math.sqrt(float(nu)/(float(nu)-2.0)) if nu > 2 else rng.standard_normal(size=shape)

    F = np.zeros((E, T), dtype=np.float32)
    for e in range(E):
        eta = t_innov(T).astype(np.float32)
        F[e, 0] = eta[0]
        for t in range(1, T):
            F[e, t] = phi_c * F[e, t-1] + math.sqrt(max(1e-8, 1.0 - phi_c**2)) * eta[t]

    eps = np.zeros((N_agents, E, T), dtype=np.float32)
    for a in range(N_agents):
        for e in range(E):
            eta = t_innov(T).astype(np.float32)
            eps[a, e, 0] = eta[0]
            for t in range(1, T):
                eps[a, e, t] = phi_e * eps[a, e, t-1] + math.sqrt(max(1e-8, 1.0 - phi_e**2)) * eta[t]

    mu = np.asarray(mu_t, dtype=np.float32)
    sd = np.asarray(std_t, dtype=np.float32)
    A = (mu[None, None, :] + sd[None, None, :] * (math.sqrt(c)*F[None, :, :] + math.sqrt(1.0-c)*eps)).transpose(0,2,1)
    # (N, T, E)
    return A


import json
# from synthetic_advantages_generator import gen_synthetic_advantages  # 基础版
# 或者：from synthetic_advantages_generator_plus import gen_adv_with_volatility as gen_synth

# 读取拟合参数
P = json.load(open("advantage_fit_params.json","r",encoding="utf-8"))
mu_t  = np.array(P["mu_t"],  dtype=np.float32)
std_t = np.array(P["std_t"], dtype=np.float32)
nu    = float(P["nu_est"])                 # ≈ 4.5
c     = float(P["avg_offdiag_corr_agents"])# ≈ 0.101
phi_c = float(P["acf1_common_factor"])     # ≈ 0.847
phi_e = float(P["acf1_series_mean"])       # ≈ 0.752
T, E  = len(mu_t), 128

# 生成 25/49/100 架无人机（形状为 (N, T, E)；如果需要和原数据一致再在末尾加一个轴 size=1）
A25  = gen_synthetic_advantages(25, T, E, nu, mu_t, std_t, c, phi_c, phi_e, seed=1)
# A49  = gen_synthetic_advantages(49, T, E, nu, mu_t, std_t, c, phi_c, phi_e, seed=2)
# A100 = gen_synthetic_advantages(100,T, E, nu, mu_t, std_t, c, phi_c, phi_e, seed=3)

np.save("synthetic_A25.npy", A25[..., None])

# # 若你希望“更重的尾部”，使用加强版（保持均值/方差曲线，但峰度≈原始16）：
# from synthetic_advantages_generator_plus import gen_adv_with_volatility
# A100_heavy = gen_adv_with_volatility(100, T, E, mu_t, std_t, c, phi_c, phi_e, kurt_target=P["kurtosis"], seed=11)
#
# # 如果你需要与原始存储格式完全一致：
# A100_heavy_npy = A100_heavy[..., None]  # -> (100, 400, 128, 1)
# np.save("synthetic_A100_heavy.npy", A100_heavy_npy)