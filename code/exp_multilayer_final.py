#!/usr/bin/env python3
"""
多层堆叠实验（最终修正版）
- 二维三角网格（周期性），堆叠 L 层
- 层间连接：每个点连接到下一层相同位置及其空间邻居（半径内）
- 势能拟合 φ ~ a r^{-α} + b，得 q = α + 1
- 预期：L=1 时 q≈1；L 增大时 q 单调上升，向 2 收敛
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from scipy.spatial import cKDTree
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import os

os.environ["OMP_NUM_THREADS"] = "32"

def build_2d_triangular_periodic(N, a=1.0):
    """生成周期性二维三角网格及其坐标"""
    pts = []
    for i in range(N):
        for j in range(N):
            x = i * a
            y = j * a * np.sqrt(3)/2
            if j % 2 == 1:
                x += a/2
            pts.append([x, y])
    pts = np.array(pts)
    box_w = N * a
    box_h = N * a * np.sqrt(3)/2
    return pts, box_w, box_h

def build_temporal_graph(layers, coords, n_side, w_time=1.0, w_space=1.0):
    N_per = len(coords)
    total_nodes = layers * N_per
    row, col, data = [], [], []
    deg = np.zeros(total_nodes)

    # 空间边界
    box_w, box_h = n_side, n_side * np.sqrt(3)/2

    # 空间边（每层内部）
    for l in range(layers):
        off = l * N_per
        tree = cKDTree(coords, boxsize=[box_w, box_h])
        edges_2d = list(tree.query_pairs(r=1.05))
        for u, v in edges_2d:
            row.extend([off+u, off+v])
            col.extend([off+v, off+u])
            data.extend([-w_space, -w_space])
            deg[off+u] += w_space
            deg[off+v] += w_space

    # 时间边（相邻层间）
    for l in range(layers - 1):
        off_c = l * N_per
        off_n = (l+1) * N_per
        tree = cKDTree(coords, boxsize=[box_w, box_h])
        # 每个节点连接到下一层对应点及其空间邻居（半径 1.2 内）
        for i in range(N_per):
            p = coords[i]
            # 查询下一层中距离 < 1.2 的所有点（周期性）
            neigh = tree.query_ball_point(p, 1.2)
            for j in neigh:
                row.extend([off_c+i, off_n+j])
                col.extend([off_n+j, off_c+i])
                data.extend([-w_time, -w_time])
                deg[off_c+i] += w_time
                deg[off_n+j] += w_time

    for i in range(total_nodes):
        row.append(i); col.append(i); data.append(deg[i])
    return sp.csr_matrix((data, (row, col)), shape=(total_nodes, total_nodes))

def solve_potential(Lmat, source_idx):
    b = np.zeros(Lmat.shape[0])
    b[source_idx] = 1.0
    phi, _ = sla.lsqr(Lmat, b, atol=1e-10, btol=1e-10)[:2]
    # 减去中位数消除常数偏移
    return phi - np.median(phi)

def compute_q(phi_layer, coords, source_pt):
    """计算单层势能的力指数 q"""
    r = np.linalg.norm(coords - source_pt, axis=1)
    # 势能归一化
    phi = phi_layer - np.median(phi_layer)
    # 远场平均剔除
    r_max = r.max()
    far = r > 0.7 * r_max
    if far.any():
        phi = phi - np.mean(phi[far])
    # 选取中场区间
    mask = (r > 0.15 * r_max) & (r < 0.55 * r_max)
    r_fit = r[mask]
    phi_fit = phi[mask]
    if len(r_fit) < 20:
        return np.nan
    # 幂律拟合
    def model(x, a, alpha, b):
        return a * x**(-alpha) + b
    try:
        popt, _ = curve_fit(model, r_fit, phi_fit, p0=[1.0, 0.1, 0.0], maxfev=5000)
        return popt[1] + 1
    except:
        return np.nan

def multilayer_experiment(n_side=40, layer_list=[1,2,4,8,12,16], w_time=1.0, seeds=3):
    print("="*60)
    print("多层堆叠实验：q vs 层数 L")
    print("="*60)
    coords, _, _ = build_2d_triangular_periodic(n_side)
    N_per = len(coords)
    source_pt = coords[N_per//2]
    results = []
    for L in layer_list:
        q_vals = []
        for seed in range(seeds):
            np.random.seed(seed)
            # 添加微小噪声
            coords_noise = coords + np.random.normal(0, 0.02, coords.shape)
            # 周期性取模
            box_w, box_h = n_side, n_side * np.sqrt(3)/2
            coords_noise = np.mod(coords_noise, [box_w, box_h])
            Lmat = build_temporal_graph(L, coords_noise, n_side, w_time=w_time)
            source_idx = (L-1) * N_per + (N_per//2)  # 顶层中心作为源
            phi_full = solve_potential(Lmat, source_idx)
            # 测量最底层（layer 0）的势能
            phi_layer = phi_full[0:N_per]
            q = compute_q(phi_layer, coords_noise, source_pt)
            q_vals.append(q)
        avg = np.mean(q_vals)
        std = np.std(q_vals)
        results.append((L, avg, std))
        print(f"L={L}: q = {avg:.3f} ± {std:.3f}")
    return results

if __name__ == "__main__":
    res = multilayer_experiment(n_side=40, layer_list=[1,2,4,8,12,16])
    Ls = [r[0] for r in res]
    q_avg = [r[1] for r in res]
    q_std = [r[2] for r in res]
    plt.errorbar(Ls, q_avg, yerr=q_std, fmt='o-', capsize=3)
    plt.xlabel('层数 L')
    plt.ylabel('力指数 q')
    plt.axhline(y=1, color='r', linestyle='--', label='2D limit (q=1)')
    plt.axhline(y=2, color='g', linestyle='--', label='3D limit (q=2)')
    plt.legend()
    plt.grid(True)
    plt.savefig('multilayer_q_vs_L.png', dpi=150)
    plt.show()