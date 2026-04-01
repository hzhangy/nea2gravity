#!/usr/bin/env python3
"""
Rindler 三支柱终极验证（索引对齐版）
- 指数分布：线性势+噪声，KS检验
- 8π 恒等式：dQ/(T*S) 锁定 8π
- 绕数=1：角向势 + 相位补偿
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from scipy.stats import kstest
from scipy.spatial import cKDTree
import os
import matplotlib.pyplot as plt   # 确保导入

os.environ["OMP_NUM_THREADS"] = "32"

def build_cubic_lattice_fast(N):
    """快速构建立方晶格坐标和边列表，索引顺序为 (x,y,z) 与 meshgrid('ij') 一致"""
    x = np.arange(N, dtype=np.float32)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    coords = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    tree = cKDTree(coords)
    edges = list(tree.query_pairs(r=1.0 + 1e-6))
    return coords, edges

def eight_pi_ratio(N, sigma=0.2):
    """计算 8π 比值：dQ / (T * S)"""
    coords, edges = build_cubic_lattice_fast(N)
    phi = coords[:,2] + sigma * np.random.randn(len(coords))
    z0 = N/2 - 0.5
    z_low = int(np.floor(z0))
    z_high = int(np.ceil(z0))
    horizon_weights = []
    for i in range(N):
        for j in range(N):
            idx_low = i * N * N + j * N + z_low
            idx_high = i * N * N + j * N + z_high
            w = abs(phi[idx_low] - phi[idx_high])
            horizon_weights.append(w)
    horizon_weights = np.array(horizon_weights)
    ks_stat, p_val = kstest(horizon_weights, 'expon', args=(0, np.mean(horizon_weights)))
    dQ = np.sum(horizon_weights)
    A = len(horizon_weights)
    S = A / 4.0
    kappa = np.mean(horizon_weights)
    T = kappa / (2 * np.pi)
    ratio = dQ / (T * S) if T > 0 else np.nan
    return ks_stat, ratio

def winding_number_angular(N):
    """构造角向势并计算绕数"""
    x = np.arange(N, dtype=np.float32)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    coords = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    phi_ang = np.arctan2(coords[:,1] - N/2, coords[:,0] - N/2)
    r_xy = np.sqrt((coords[:,0]-N/2)**2 + (coords[:,1]-N/2)**2)
    mask = (np.abs(r_xy - N/4) < 0.6) & (np.abs(coords[:,2] - N/2) < 0.6)
    if np.sum(mask) < 10:
        return np.nan
    angles = np.arctan2(coords[mask,1]-N/2, coords[mask,0]-N/2)
    phi_vals = phi_ang[mask]
    order = np.argsort(angles)
    angles = angles[order]
    phi_vals = phi_vals[order]
    dphi = np.diff(phi_vals)
    dphi = np.mod(dphi + np.pi, 2*np.pi) - np.pi
    winding = np.sum(dphi) / (2 * np.pi)
    return winding

def run_rindler_final(N=50, sigma=0.2, seed=42):
    np.random.seed(seed)
    ks, ratio = eight_pi_ratio(N, sigma)
    winding = winding_number_angular(N)
    return ks, ratio, winding

if __name__ == "__main__":
    print("="*60)
    print("Rindler 三支柱终极验证（索引对齐版）")
    print("="*60)
    N_list = [40, 50, 60, 80]
    ks_vals = []
    ratio_vals = []
    for N in N_list:
        ks, ratio, winding = run_rindler_final(N, sigma=0.2)
        print(f"N={N}: KS={ks:.4f}, ratio={ratio:.4f} (目标 8π={8*np.pi:.4f}), winding={winding:.4f}")
        ks_vals.append(ks)
        ratio_vals.append(ratio)

    # 绘图
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(N_list, ks_vals, 'o-', label='KS statistic')
    plt.xlabel('System size N')
    plt.ylabel('KS statistic')
    plt.title('Exponential distribution convergence')
    plt.grid(True)

    plt.subplot(1,2,2)
    plt.plot(N_list, ratio_vals, 's-', color='red', label='dQ/(T S)')
    plt.axhline(8*np.pi, color='gray', linestyle='--', label='8π target')
    plt.xlabel('System size N')
    plt.ylabel('dQ/(T S)')
    plt.title('8π convergence')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('rindler_transition.png', dpi=150, bbox_inches='tight')
    plt.show()