"""
终极盲测增强版：增加数据点，更稳健的拟合
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

# ========== 1. 读取数据 ==========
col_names = ['Galaxy', 'R', 'Vobs', 'Verr', 'Vgas', 'Vdisk', 'Vbulge', 'Vhalo', 'Flag']
raw = pd.read_csv('table2.txt', sep=r'\s+', names=col_names)
raw['Galaxy'] = raw['Galaxy'].astype(str).str.strip()

targets = {
    'DDO154': '4.04',
    'UGC128': '64.5',
    'F583-1': '35.4'
}

# 提取数据
gal_data = {}
for name, num in targets.items():
    g = raw[raw['Galaxy'] == num].sort_values('R')
    if len(g) == 0:
        print(f"未找到星系 {name} (编号 {num})")
        continue
    gal_data[name] = g

if not gal_data:
    print("没有找到目标星系")
    exit()

G_kpc = 4.302e-6
Sigma_crit_kpc2 = 194 * 1e6
M_L_disk = 0.5
M_L_bulge = 0.7

def v_model(r, v0, q):
    return v0 * r ** ((1 - q) / 2)

print(f"{'Galaxy':<10} {'拟合点数':>6} {'观测 q_obs':>12} {'预测 q_pred':>12} {'差值':>8}")
print("-" * 55)

for name, g in gal_data.items():
    # 取最后 6 个点（或全部外围点）
    n_points = min(6, len(g))
    data = g.tail(n_points)
    R = data['R'].values
    Vobs = data['Vobs'].values
    Verr = data['Verr'].values
    Verr = np.maximum(Verr, 0.1)
    
    # 观测 q_obs
    try:
        popt, pcov = curve_fit(v_model, R, Vobs, sigma=Verr, p0=[Vobs.mean(), 1.0])
        q_obs = popt[1]
        q_err = np.sqrt(pcov[1,1]) if pcov is not None else 0
    except Exception as e:
        print(f"{name:<10} 拟合失败: {e}")
        continue
    
    # 理论预测 q_pred
    Vb = np.sqrt(data['Vgas']**2 + M_L_disk * data['Vdisk']**2 + M_L_bulge * data['Vbulge']**2).values
    Mbar = Vb**2 * R / G_kpc
    Sigma = Mbar / (np.pi * R**2 + 1e-10)
    q_pred = 1 + np.tanh(Sigma / Sigma_crit_kpc2)
    q_pred_mean = np.mean(q_pred)
    
    diff = abs(q_obs - q_pred_mean)
    print(f"{name:<10} {len(R):6d} {q_obs:12.3f} ± {q_err:.3f} {q_pred_mean:12.3f} {diff:8.4f}")

# 额外打印每个星系的详细拟合点，以便检查
print("\n" + "="*60)
print("详细数据点")
for name, g in gal_data.items():
    print(f"\n{name}:")
    data = g.tail(6)
    R = data['R'].values
    Vobs = data['Vobs'].values
    Verr = data['Verr'].values
    print(f"{'R(kpc)':>8} {'Vobs':>8} {'Verr':>8} {'Vgas':>8} {'Vdisk':>8} {'Vbulge':>8}")
    for i in range(len(R)):
        print(f"{R[i]:8.2f} {Vobs[i]:8.1f} {Verr[i]:8.1f} {data['Vgas'].values[i]:8.1f} {data['Vdisk'].values[i]:8.1f} {data['Vbulge'].values[i]:8.1f}")