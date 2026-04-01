#!/usr/bin/env python3
"""
实验 A（修正版）：二维引力确认（支持不同尺度）
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from scipy.spatial import cKDTree
from scipy.optimize import curve_fit
import os

os.environ["OMP_NUM_THREADS"] = "32"

def build_2d_triangular_periodic(N, a=1.0):
    pts = []
    for i in range(N):
        for j in range(N):
            x = i * a
            y = j * a * np.sqrt(3)/2
            if j % 2 == 1:
                x += a/2
            pts.append([x, y])
    pts = np.array(pts)
    tree = cKDTree(pts, boxsize=[N*a, N*a*np.sqrt(3)/2])
    edges = list(tree.query_pairs(r=a+1e-6))
    return pts, edges

def build_rgg_periodic(N, avg_deg=400):
    def torus_dist(p1, p2):
        delta = np.abs(p1 - p2)
        delta = np.minimum(delta, 1 - delta)
        return np.sqrt(np.sum(delta**2))
    r = (3 * avg_deg / (4 * np.pi * N))**(1/3)
    pts = np.random.rand(N, 3)
    edges = set()
    for i in range(N):
        for j in range(i+1, N):
            if torus_dist(pts[i], pts[j]) < r:
                edges.add((i, j))
    return pts, list(edges)

def laplacian_from_edges(N, edges):
    row, col, data = [], [], []
    deg = np.zeros(N, dtype=int)
    for u, v in edges:
        row.extend([u, v]); col.extend([v, u]); data.extend([-1, -1])
        deg[u] += 1; deg[v] += 1
    for i in range(N):
        row.append(i); col.append(i); data.append(deg[i])
    return sp.csr_matrix((data, (row, col)), shape=(N, N))

def solve_potential(L, src):
    b = np.zeros(L.shape[0])
    b[src] = 1.0
    phi, _ = sla.lsqr(L, b, atol=1e-10, btol=1e-10)[:2]
    return phi - np.median(phi)

def compute_q_fixed(phi, pts, src, rmin, rmax):
    src_pt = pts[src]
    r = np.linalg.norm(pts - src_pt, axis=1)
    mask = (r > rmin) & (r < rmax)
    r_fit = r[mask]
    phi_fit = phi[mask]
    if len(r_fit) < 20:
        return np.nan, np.nan, np.nan, np.nan

    def log_model(x, a, b): return a * np.log(x) + b
    def inv_model(x, a, b): return a / x + b

    try:
        popt_log, _ = curve_fit(log_model, r_fit, phi_fit, p0=[-1.0, 0.0], maxfev=5000)
        resid_log = np.sum((phi_fit - log_model(r_fit, *popt_log))**2)
        popt_inv, _ = curve_fit(inv_model, r_fit, phi_fit, p0=[1.0, 0.0], maxfev=5000)
        resid_inv = np.sum((phi_fit - inv_model(r_fit, *popt_inv))**2)
        def power_law(x, a, alpha, b): return a * x**(-alpha) + b
        popt_pow, _ = curve_fit(power_law, r_fit, phi_fit, p0=[1.0, 0.1, 0.0], maxfev=5000)
        alpha = popt_pow[1]
        q_free = alpha + 1
        q_model = 1 if resid_log < resid_inv else 2
        return q_model, q_free, resid_log, resid_inv
    except:
        return np.nan, np.nan, np.nan, np.nan

def main():
    print("="*60)
    print("实验 A（修正版）：二维引力确认")
    print("="*60)

    # 2D 三角网格
    print("\n构建 2D 三角网格 (200x200, 约 40k 节点)...")
    pts2d, edges2d = build_2d_triangular_periodic(200)
    L2d = laplacian_from_edges(len(pts2d), edges2d)
    src = len(pts2d)//2
    print("求解势能...")
    phi2d = solve_potential(L2d, src)
    print("拟合力指数...")
    q_model, q_free, res_log, res_inv = compute_q_fixed(phi2d, pts2d, src, rmin=5.0, rmax=25.0)
    print(f"2D 三角网格 (200x200, 40k节点): model q={q_model}, free q={q_free:.3f}, "
          f"log resid={res_log:.3e}, inv resid={res_inv:.3e}")

    # 3D RGG 周期性
    print("\n构建 3D RGG (N=4000, avg_deg=400)...")
    pts3d, edges3d = build_rgg_periodic(4000, 400)
    L3d = laplacian_from_edges(len(pts3d), edges3d)
    src = len(pts3d)//2
    print("求解势能...")
    phi3d = solve_potential(L3d, src)
    print("拟合力指数...")
    q_model, q_free, res_log, res_inv = compute_q_fixed(phi3d, pts3d, src, rmin=0.08, rmax=0.35)
    print(f"3D RGG (N=4000, deg=400): model q={q_model}, free q={q_free:.3f}, "
          f"log resid={res_log:.3e}, inv resid={res_inv:.3e}")

if __name__ == "__main__":
    main()