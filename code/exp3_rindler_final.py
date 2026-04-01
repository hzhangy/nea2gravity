#!/usr/bin/env python3
"""
实验3：稀疏图极限（Lloyd 松弛 + 周期性边界）
- 随机点 → 松弛 → 均匀网格 → 引力指数 q → 1
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from scipy.spatial import cKDTree
from scipy.optimize import curve_fit
import os
import warnings
warnings.filterwarnings('ignore')

os.environ["OMP_NUM_THREADS"] = "32"

def generate_random_points(N, dim=2, boxsize=1.0):
    """生成周期性边界内的随机点"""
    return np.random.rand(N, dim) * boxsize

def lloyd_relax_periodic(points, k_neighbors=8, boxsize=1.0, relax_steps=1):
    """
    单步 Lloyd 松弛，使用周期性 cKDTree 和 k‑NN 邻居
    返回新点集和边列表（用于拉普拉斯）
    """
    for _ in range(relax_steps):
        tree = cKDTree(points, boxsize=boxsize)
        # 查询每个点的 k 个最近邻（包括自身）
        dists, idxs = tree.query(points, k=k_neighbors+1)
        new_points = np.zeros_like(points)
        for i in range(len(points)):
            # 邻居索引（排除自身）
            neigh_idx = idxs[i, 1:]
            # 邻居坐标
            neigh_pts = points[neigh_idx]
            # 周期性修正：将邻居坐标平移到 i 点的最近副本
            delta = neigh_pts - points[i]
            delta = delta - np.round(delta / boxsize) * boxsize
            corrected = points[i] + delta
            new_points[i] = np.mean(corrected, axis=0)
        points = np.mod(new_points, boxsize)  # 限制在盒子内
    # 最终重新连接边（用于求解势能）
    tree = cKDTree(points, boxsize=boxsize)
    # 连接半径取使得平均度约为 10 的适当值（这里固定 0.15，可根据情况调整）
    edges = list(tree.query_pairs(r=0.15))
    return points, edges

def laplacian_from_edges(N_nodes, edges):
    row, col, data = [], [], []
    deg = np.zeros(N_nodes, dtype=int)
    for u, v in edges:
        row.append(u); col.append(v); data.append(-1)
        row.append(v); col.append(u); data.append(-1)
        deg[u] += 1
        deg[v] += 1
    for i in range(N_nodes):
        row.append(i); col.append(i); data.append(deg[i])
    L = sp.coo_matrix((data, (row, col)), shape=(N_nodes, N_nodes))
    return L.tocsr()

def solve_potential(L, source_idx):
    b = np.zeros(L.shape[0])
    b[source_idx] = 1.0
    phi, info = sla.lsqr(L, b, atol=1e-10, btol=1e-10)[:2]
    return phi - np.median(phi)   # 归一化

def compute_q_free(phi, points, source_idx):
    source = points[source_idx]
    r = np.linalg.norm(points - source, axis=1)
    # 势能归一化：减中位数，再减远场平均
    phi = phi - np.median(phi)
    r_max = r.max()
    far = r > 0.7 * r_max
    if far.any():
        phi = phi - np.mean(phi[far])
    # 过滤源点自身
    mask = r > 0
    r, phi = r[mask], phi[mask]
    # 拟合区间：基于排序距离的 15%–55%
    r_sorted = np.sort(r)
    inner = r_sorted[int(0.15 * len(r_sorted))]
    outer = r_sorted[int(0.55 * len(r_sorted))]
    fit_mask = (r >= inner) & (r <= outer)
    r_fit, phi_fit = r[fit_mask], phi[fit_mask]
    if len(r_fit) < 20:
        return np.nan
    def power_law(x, a, alpha, b):
        return a * x**(-alpha) + b
    try:
        popt, _ = curve_fit(power_law, r_fit, phi_fit, p0=[1.0, 0.1, 0.0], maxfev=5000)
        alpha = popt[1]
        q = alpha + 1
        return q
    except:
        return np.nan

def experiment3(N_list=[500, 1000, 2000], relax_total=6, seeds=3):
    print("="*60)
    print("实验3：稀疏图极限（Lloyd 松弛 + 周期性边界）")
    print("="*60)
    all_results = {}
    for N in N_list:
        print(f"\nN = {N}")
        q_finals = []
        for seed in range(seeds):
            np.random.seed(seed)
            points = generate_random_points(N)
            # 执行全部松弛步骤
            for step in range(relax_total):
                points, edges = lloyd_relax_periodic(points, k_neighbors=8, relax_steps=1)
            # 最终求解势能
            L = laplacian_from_edges(N, edges)
            source = N // 2
            phi = solve_potential(L, source)
            q = compute_q_free(phi, points, source)
            q_finals.append(q)
            print(f"  seed={seed}: final q = {q:.3f}")
        avg_q = np.mean(q_finals)
        std_q = np.std(q_finals)
        print(f"  average q = {avg_q:.3f} ± {std_q:.3f}")
        all_results[N] = {'q': q_finals, 'avg': avg_q, 'std': std_q}
    return all_results

if __name__ == "__main__":
    results = experiment3()