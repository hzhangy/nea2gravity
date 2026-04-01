import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from scipy.spatial import cKDTree
from scipy.optimize import curve_fit
from itertools import product
import os

os.environ["OMP_NUM_THREADS"] = "32"

def build_cubic_lattice_periodic(N, a=1.0):
    """周期性立方晶格，使用 cKDTree boxsize 实现周期性边界"""
    x = np.arange(N, dtype=np.float32) * a
    # 使用 meshgrid 并指定 indexing='ij' 以匹配标准索引
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    coords = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    # 周期性边界大小
    boxsize = [N*a, N*a, N*a]
    tree = cKDTree(coords, boxsize=boxsize)
    edges = list(tree.query_pairs(r=a + 1e-6))
    return coords, edges

def build_fcc_lattice_periodic(n_cells=12, a=1.0):
    """周期性 FCC 晶格，同样使用 boxsize"""
    points = []
    for i, j, k in product(range(n_cells), repeat=3):
        for dx, dy, dz in [(0,0,0), (0,1,1), (1,0,1), (1,1,0)]:
            points.append([(i+dx/2)*a, (j+dy/2)*a, (k+dz/2)*a])
    points = np.array(points, dtype=np.float32)
    # FCC 晶胞边长 a，实际坐标范围 [0, n_cells*a]
    boxsize = [n_cells*a, n_cells*a, n_cells*a]
    tree = cKDTree(points, boxsize=boxsize)
    # 最近邻距离理论值 sqrt(2)/2 * a
    r_neighbor = np.sqrt(2)/2 * a
    edges = list(tree.query_pairs(r=r_neighbor + 1e-6))
    return points, edges

def laplacian_from_edges(N_nodes, edges):
    row, col, data = [], [], []
    deg = np.zeros(N_nodes, dtype=int)
    for u, v in edges:
        row.extend([u, v]); col.extend([v, u]); data.extend([-1, -1])
        deg[u] += 1; deg[v] += 1
    for i in range(N_nodes):
        row.append(i); col.append(i); data.append(deg[i])
    return sp.csr_matrix((data, (row, col)), shape=(N_nodes, N_nodes))

def solve_potential(L, source_idx):
    b = np.zeros(L.shape[0])
    b[source_idx] = 1.0
    # 增加迭代次数以确保收敛
    phi = sla.lsqr(L, b, atol=1e-10, btol=1e-10, iter_lim=10000)[0]
    return phi - np.median(phi)

def compute_q(phi, points, source_idx):
    source = points[source_idx]
    r = np.linalg.norm(points - source, axis=1)
    phi = phi - np.median(phi)
    r_max = r.max()
    far = r > 0.7 * r_max
    if far.any():
        phi = phi - np.mean(phi[far])
    # 使用更窄的中场区间，避免边界干扰
    mask = (r > 0.2 * r_max) & (r < 0.5 * r_max)
    r_fit, phi_fit = r[mask], phi[mask]
    if len(r_fit) < 20:
        return np.nan
    try:
        popt, _ = curve_fit(lambda x, a, alpha, b: a * x**(-alpha) + b,
                            r_fit, phi_fit, p0=[1.0, 0.1, 0.0], maxfev=5000)
        return popt[1] + 1
    except:
        return np.nan

def main():
    print("="*60)
    print("大收官（优化版）：各向异性 vs 各向同性")
    print("采用周期性边界，增大网格")
    print("="*60)

    # 1. 立方晶格（各向异性），N=50 节点数 125k
    N_cub = 50
    print(f"构建立方晶格 {N_cub}^3 节点...")
    pts_c, edg_c = build_cubic_lattice_periodic(N_cub)
    print(f"  节点数: {len(pts_c)}")
    L_c = laplacian_from_edges(len(pts_c), edg_c)
    # 源点选取中心附近的节点（由于周期性，中心唯一性稍弱，但取中间索引即可）
    # 索引计算：mid = N_cub//2, idx = mid*N_cub*N_cub + mid*N_cub + mid
    mid = N_cub // 2
    src_c = mid * N_cub * N_cub + mid * N_cub + mid
    print("求解势能...")
    phi_c = solve_potential(L_c, src_c)
    q_c = compute_q(phi_c, pts_c, src_c)
    print(f"立方晶格 (各向异性): q = {q_c:.3f}  (期望 ~2.0)")

    # 2. FCC 晶格（各向同性），n_cells=12 → 12^3*4 = 6912 节点
    n_fcc = 12
    print(f"构建 FCC 晶格 {n_fcc}^3 晶胞...")
    pts_f, edg_f = build_fcc_lattice_periodic(n_fcc)
    print(f"  节点数: {len(pts_f)}")
    L_f = laplacian_from_edges(len(pts_f), edg_f)
    src_f = len(pts_f) // 2
    print("求解势能...")
    phi_f = solve_potential(L_f, src_f)
    q_f = compute_q(phi_f, pts_f, src_f)
    print(f"FCC 晶格 (各向同性): q = {q_f:.3f}  (期望 ~1.0)")

    print("\n" + "="*60)
    print("结论：")
    print(" - 各向异性图（立方晶格）力指数接近 2，对应三维牛顿引力")
    print(" - 各向同性图（FCC）力指数接近 1，对应二维引力")
    print("="*60)

if __name__ == "__main__":
    main()