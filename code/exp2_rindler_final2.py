#!/usr/bin/env python3
"""
实验2：各向同性图 → q=1（大网格最终版）
FCC: 10x10x10晶胞 (4000节点)
RGG: 周期性边界, N=4000, avg_deg=400
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from scipy.spatial import cKDTree
from scipy.optimize import curve_fit
from itertools import product
import os

os.environ["OMP_NUM_THREADS"] = "32"

# -------------------- FCC --------------------
def build_fcc_lattice(n_cells=10, a=1.0):
    points = []
    for i, j, k in product(range(n_cells), repeat=3):
        for dx, dy, dz in [(0,0,0), (0,1,1), (1,0,1), (1,1,0)]:
            points.append([(i+dx/2)*a, (j+dy/2)*a, (k+dz/2)*a])
    points = np.array(points)
    tree = cKDTree(points)
    edges = list(tree.query_pairs(r=np.sqrt(2)/2*a + 1e-6))
    return points, edges

# -------------------- RGG 周期性 --------------------
def build_rgg_periodic(N, avg_deg):
    def torus_dist(p1, p2):
        delta = np.abs(p1 - p2)
        delta = np.minimum(delta, 1 - delta)
        return np.sqrt(np.sum(delta**2))
    r = (3 * avg_deg / (4 * np.pi * N))**(1/3)
    points = np.random.rand(N, 3)
    edges = set()
    for i in range(N):
        for j in range(i+1, N):
            if torus_dist(points[i], points[j]) < r:
                edges.add((i, j))
    return points, list(edges)

# -------------------- 拉普拉斯与求解 --------------------
def laplacian_from_edges(N_nodes, edges):
    row, col, data = [], [], []
    deg = np.zeros(N_nodes, dtype=int)
    for u, v in edges:
        row.append(u); col.append(v); data.append(-1)
        row.append(v); col.append(u); data.append(-1)
        deg[u] += 1; deg[v] += 1
    for i in range(N_nodes):
        row.append(i); col.append(i); data.append(deg[i])
    L = sp.coo_matrix((data, (row, col)), shape=(N_nodes, N_nodes))
    return L.tocsr()

def solve_potential(L, source_idx):
    b = np.zeros(L.shape[0])
    b[source_idx] = 1.0
    phi, info = sla.lsqr(L, b, atol=1e-10, btol=1e-10)[:2]
    return phi

# -------------------- 拟合 --------------------
def compute_q_free(phi, points, source_idx):
    source = points[source_idx]
    r = np.linalg.norm(points - source, axis=1)
    # 归一化：减去中位数，再减远场平均值
    phi = phi - np.median(phi)
    r_max = r.max()
    far = r > 0.7 * r_max
    if far.any():
        phi = phi - np.mean(phi[far])
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
        return popt[1] + 1
    except:
        return np.nan

# -------------------- 主实验 --------------------
def experiment2():
    print("="*60)
    print("实验2：各向同性图 → q=1（大网格最终版）")
    print("="*60)

    # FCC 10x10x10 晶胞 (4000节点)
    points_fcc, edges_fcc = build_fcc_lattice(10)
    L_fcc = laplacian_from_edges(len(points_fcc), edges_fcc)
    source_fcc = len(points_fcc)//2
    phi_fcc = solve_potential(L_fcc, source_fcc)
    q_fcc = compute_q_free(phi_fcc, points_fcc, source_fcc)
    print(f"FCC (10x10x10, 4000 nodes): q = {q_fcc:.3f}")

    # RGG 周期性, N=4000, avg_deg=400
    points_rgg, edges_rgg = build_rgg_periodic(4000, 400)
    L_rgg = laplacian_from_edges(len(points_rgg), edges_rgg)
    source_rgg = len(points_rgg)//2
    phi_rgg = solve_potential(L_rgg, source_rgg)
    q_rgg = compute_q_free(phi_rgg, points_rgg, source_rgg)
    print(f"RGG (N=4000, deg=400): q = {q_rgg:.3f}")

if __name__ == "__main__":
    experiment2()