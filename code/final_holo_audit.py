import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# ========== 1. 按字节解析 Table1.mrt ==========
def parse_table1_fixed(filename):
    gal_data = {}
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(('Name', '---', 'Byte')):
                continue
            parts = line.split()
            if len(parts) < 10:
                continue
            name = parts[0]
            if name[0].isdigit():
                continue
            try:
                dist = float(parts[1])
                lum = float(parts[4])
                reff = float(parts[5])
                mhi = float(parts[9])
                gal_data[name] = {'D': dist, 'L36': lum*1e9, 'Reff': reff, 'MHI': mhi*1e9}
            except:
                continue
    return gal_data

# ========== 2. 读取旋转曲线数据并映射星系名 ==========
def load_rotation_curves():
    # table2.txt 第一列是序号，与 Table1 顺序对应
    raw = pd.read_csv('table2.txt', sep=r'\s+', 
                      names=['idx', 'R', 'Vobs', 'Verr', 'Vgas', 'Vdisk', 'Vbulge', 'Vhalo', 'Flag'])
    # 将 idx 转为整数
    raw['idx'] = pd.to_numeric(raw['idx'], errors='coerce').fillna(0).astype(int)
    # 从 Table1 获取序号-星系名映射
    t1_map = parse_table1_fixed('Table1.mrt')
    # 构建 idx -> name 的映射（Table1 的 idx 从 1 开始，与 table2 的 idx 对应）
    idx_to_name = {}
    for i, (name, params) in enumerate(t1_map.items(), start=1):
        idx_to_name[i] = name
    # 映射到 table2
    raw['Galaxy'] = raw['idx'].map(idx_to_name)
    # 剔除无法映射的行（idx 可能超出范围）
    raw = raw.dropna(subset=['Galaxy'])
    return raw, t1_map

# ========== 3. 主审计流程 ==========
def final_audit():
    print("正在解析 Table1.mrt...")
    t1_map = parse_table1_fixed('Table1.mrt')
    print(f"成功解析 {len(t1_map)} 个星系")

    print("正在加载旋转曲线...")
    raw, _ = load_rotation_curves()
    print(f"旋转曲线数据成功映射 {len(raw)} 行，{raw['Galaxy'].nunique()} 个星系")

    # 读取 N.E.A. 拟合结果
    fit = pd.read_csv('nea_hd_fit.csv')
    fit['Galaxy'] = fit['Galaxy'].astype(str).str.strip()

    # 提取整数部分作为索引（例如 "15.2" -> 15）
    fit['idx_int'] = fit['Galaxy'].str.extract(r'^(\d+)').astype('Int64')

    # 构建编号到名称的映射（使用整数索引）
    t1_map = parse_table1_fixed('Table1.mrt')
    idx_to_name = {}
    for i, (name, params) in enumerate(t1_map.items(), start=1):
        idx_to_name[i] = name

    # 将整数索引映射为星系名
    fit['GalaxyName'] = fit['idx_int'].map(idx_to_name)

    # 删除无法映射的行
    fit = fit.dropna(subset=['GalaxyName'])
    # 使用名称作为合并键
    fit['Galaxy'] = fit['GalaxyName']
    fit = fit.drop(columns=['GalaxyName'])
 
    good = fit[(fit['chi2_red'] < 5) & (fit['cov_ok'] == True)].copy()
    print(f"N.E.A. 拟合成功: {len(fit)} 星系, 质量良好: {len(good)} 星系")

    # 合并
    merged = good.merge(pd.DataFrame(t1_map).T.reset_index().rename(columns={'index': 'Galaxy'}),
                        on='Galaxy', how='inner')
    print(f"匹配到的星系数量: {len(merged)}")
    if len(merged) == 0:
        print("匹配失败，请检查星系名格式。打印双方前10个:")
        print("good 前10:", good['Galaxy'].head(10).tolist())
        print("table1 前10:", list(t1_map.keys())[:10])
        return

    # 计算面密度（恒星+气体）
    merged['Mstar'] = merged['L36'] * 0.5          # M/L=0.5
    merged['Mgas'] = merged['MHI'] * 1.33          # 含氦
    merged['Mbar'] = merged['Mstar'] + merged['Mgas']
    merged['Sigma_bar'] = merged['Mbar'] / (np.pi * merged['Reff']**2)   # Msun/kpc^2

    # 计算全息不变量
    all_inv = []
    galaxy_stats = []
    for idx, row in merged.iterrows():
        gal = row['Galaxy']
        q = row['q']
        Rc = row['Rc']
        data = raw[raw['Galaxy'] == gal].sort_values('R')
        if len(data) < 4:
            continue
        R = data['R'].values
        Vobs = data['Vobs'].values
        # 重子速度贡献（采用修正的 M/L）
        Vb = np.sqrt(data['Vgas']**2 + 0.5 * data['Vdisk']**2 + 0.7 * data['Vbulge']**2).values
        safe_Vb = np.maximum(Vb, 1e-5)
        x = R / Rc  
        factor = 1.0 + x
        inv = (Vobs / safe_Vb)**2 / (factor ** (2 - q))
        valid = (x > 0.05) & (x < 20) & (Vobs/safe_Vb > 0) & (Vobs/safe_Vb < 3) & (np.isfinite(inv))
        if not any(valid):
            continue
        all_inv.extend(inv[valid])
        galaxy_stats.append({
            'gal': gal,
            'q': q,
            'Rc': Rc,
            'Sigma_bar': row['Sigma_bar'],
            'Mstar': row['Mstar'],
            'Mgas': row['Mgas'],
            'D': row['D'],
            'Reff': row['Reff'],
            'ndata': sum(valid)
        })

    stats_df = pd.DataFrame(galaxy_stats)
    if len(stats_df) == 0:
        print("没有星系能计算不变量，退出。")
        return

    mean_inv = np.mean(all_inv)
    std_inv = np.std(all_inv)
    print(f"\n全息不变量 I 的均值: {mean_inv:.4f} (理论值 1.0000)")
    print(f"I 的标准差: {std_inv:.4f}")

    # 绘图
    plt.rcParams['font.size'] = 12
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax1, ax2, ax3, ax4 = axes.flatten()

    # 图1：不变量直方图
    ax1.hist(all_inv, bins=60, range=(0.5, 1.5), color='skyblue', edgecolor='black')
    ax1.axvline(1.0, color='red', linestyle='--', lw=2, label='Theoretical')
    ax1.set_xlabel('$I$')
    ax1.set_ylabel('Count')
    ax1.set_title(f'Invariant (mean={mean_inv:.3f}, std={std_inv:.3f})')
    ax1.legend()

    # 图2：q vs 面密度
    sc = ax2.scatter(stats_df['Sigma_bar'], stats_df['q'],
                     c=np.log10(stats_df['Rc']), cmap='viridis',
                     s=50, alpha=0.8, edgecolors='black')
    ax2.set_xscale('log')
    ax2.set_xlabel(r'$\Sigma_{\rm bar}$ ($M_\odot$/kpc$^2$)')
    ax2.set_ylabel('$q$')
    ax2.set_title('Dimension vs. Baryon Surface Density')
    cbar = plt.colorbar(sc, ax=ax2)
    cbar.set_label(r'$\log_{10}(R_c)$')
    ax2.grid(True, alpha=0.2)

    # Sigmoid 拟合
    def sigmoid(x, a, b, x0):
        return a + b / (1 + np.exp(-(np.log10(x) - x0)))
    try:
        x_fit = stats_df['Sigma_bar'].values
        y_fit = stats_df['q'].values
        mask = (x_fit > 0) & (np.isfinite(x_fit)) & (np.isfinite(y_fit))
        if sum(mask) > 3:
            popt, _ = curve_fit(sigmoid, x_fit[mask], y_fit[mask], p0=[1.5, 1.0, 2.0], maxfev=5000)
            x_sig = np.logspace(np.log10(x_fit[mask].min()), np.log10(x_fit[mask].max()), 100)
            y_sig = sigmoid(x_sig, *popt)
            ax2.plot(x_sig, y_sig, 'r--', lw=2,
                     label=f'Sigmoid: q = {popt[0]:.2f} + {popt[1]:.2f} / (1+exp(-log10Σ-{popt[2]:.2f}))')
            ax2.legend()
    except Exception as e:
        print(f"Sigmoid 拟合失败: {e}")

    # 图3：Rc vs 面密度
    ax3.scatter(stats_df['Sigma_bar'], stats_df['Rc'], s=30, alpha=0.6, c='teal')
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.set_xlabel(r'$\Sigma_{\rm bar}$ ($M_\odot$/kpc$^2$)')
    ax3.set_ylabel('$R_c$ (kpc)')
    ax3.set_title('Characteristic Scale vs. Density')
    ax3.grid(True, alpha=0.2)

    # 图4：Mstar vs Mgas
    ax4.scatter(stats_df['Mstar'], stats_df['Mgas'], s=30, alpha=0.6, c=stats_df['q'], cmap='plasma')
    ax4.set_xscale('log')
    ax4.set_yscale('log')
    ax4.set_xlabel('$M_\\star$ ($M_\\odot$)')
    ax4.set_ylabel('$M_{\\rm gas}$ ($M_\\odot$)')
    ax4.set_title('Stellar vs. Gas Mass')
    cbar = plt.colorbar(ax4.collections[0], ax=ax4)
    cbar.set_label('$q$')
    ax4.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig('final_holo_audit.png', dpi=150)
    plt.show()

    # 统计
    corr, pval = pearsonr(np.log10(stats_df['Sigma_bar']), stats_df['q'])
    print("\n" + "="*50)
    print("最终全息审计报告")
    print("="*50)
    print(f"参与审计的星系数量: {len(stats_df)}")
    print(f"全息不变量 I 的均值: {mean_inv:.4f} (理论值 1.0000)")
    print(f"I 的标准差: {std_inv:.4f}")
    if abs(mean_inv - 1.0) < 0.05:
        print("✅ 不变量验证通过！")
    else:
        print("⚠️ 不变量存在系统偏差，可能源于质量模型误差。")

    print(f"\nlog10(Σ) 与 q 的相关系数: {corr:.3f} (p = {pval:.2e})")
    if pval < 0.01:
        print("✅ 显著正相关，维度随面密度增加而增大。")
    else:
        print("⚠️ 相关性不显著，可能需要更精确的面密度计算。")

    print(f"\nq 的统计分布 (共 {len(stats_df)} 星系):")
    print(f"  q ≈ 1.0 (1.0-1.2): {sum((stats_df['q'] >= 1.0) & (stats_df['q'] < 1.2))}")
    print(f"  q ≈ 1.5 (1.4-1.6): {sum((stats_df['q'] >= 1.4) & (stats_df['q'] <= 1.6))}")
    print(f"  q ≈ 2.0 (1.8-2.0): {sum((stats_df['q'] > 1.8) & (stats_df['q'] <= 2.0))}")

    stats_df.to_csv('nea_precise_audit.csv', index=False)
    print("\n详细结果已保存至 nea_precise_audit.csv")

if __name__ == "__main__":
    final_audit()