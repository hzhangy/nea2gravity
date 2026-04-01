import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import warnings

# 读取数据
col_names = ['Galaxy', 'R', 'Vobs', 'Verr', 'Vgas', 'Vdisk', 'Vbulge', 'Vhalo', 'Flag']
df = pd.read_csv('table2.txt', sep=r'\s+', names=col_names, comment=None, header=None)

# 误差处理：零误差替换为观测速度的1%或0.1
df['Verr'] = df['Verr'].replace(0, np.nan).fillna(0.1)
df['Verr'] = df.apply(lambda row: max(row['Verr'], 0.01*row['Vobs']), axis=1)

def nea_model(R, q, Rc, Vbaryon):
    """N.E.A. 全息维度模型"""
    term = (R / Rc) ** (2 - q)
    return Vbaryon * np.sqrt(1 + term)

results = []

for gal in df['Galaxy'].unique():
    g = df[df['Galaxy'] == gal].sort_values('R')
    R = g['R'].values
    Vobs = g['Vobs'].values
    Verr = g['Verr'].values
    Vb = np.sqrt(g['Vgas']**2 + g['Vdisk']**2 + g['Vbulge']**2).values

    if len(R) < 5:   # 至少5个点才能可靠拟合两个参数
        continue

    def model_to_fit(R_val, q, Rc):
        return nea_model(R_val, q, Rc, Vb)

    try:
        popt, pcov = curve_fit(model_to_fit, R, Vobs, sigma=Verr,
                               p0=[1.5, np.median(R)],  # 初始猜测
                               bounds=([1.0, 0.01], [2.0, 500.0]),
                               maxfev=5000)
        q_fit, Rc_fit = popt
        chi2 = np.sum(((Vobs - model_to_fit(R, q_fit, Rc_fit)) / Verr)**2)
        chi2_red = chi2 / (len(R) - 2)
        cov_ok = not (np.isinf(pcov).any() or np.isnan(pcov).any())
        results.append({
            'Galaxy': gal,
            'q': q_fit,
            'Rc': Rc_fit,
            'chi2_red': chi2_red,
            'cov_ok': cov_ok,
            'ndata': len(R)
        })
        print(f"{gal}: q = {q_fit:.3f}, Rc = {Rc_fit:.2f}, χ²/ν = {chi2_red:.2f}")
    except Exception as e:
        print(f"{gal}: 拟合失败 - {e}")

# 分类统计
good = [r for r in results if r['cov_ok'] and r['chi2_red'] < 5]
q1 = [r for r in good if 1.0 <= r['q'] <= 1.2]
q15 = [r for r in good if 1.4 <= r['q'] <= 1.6]
q2 = [r for r in good if 1.8 <= r['q'] <= 2.0]

print(f"\n成功拟合且质量良好（χ²/ν<5）: {len(good)} 个星系")
print(f"  q ≈ 1 类 (1.0-1.2): {len(q1)} 个")
print(f"  q ≈ 1.5 类 (1.4-1.6): {len(q15)} 个")
print(f"  q ≈ 2 类 (1.8-2.0): {len(q2)} 个")

import matplotlib.pyplot as plt

# 提取质量良好拟合的 q 值
q_vals = [r['q'] for r in good]

plt.figure(figsize=(8,5))
plt.hist(q_vals, bins=20, edgecolor='black', alpha=0.7)
plt.xlabel('q')
plt.ylabel('Number of galaxies')
plt.title('Distribution of effective gravitational dimension q')
plt.grid(True, alpha=0.3)
plt.savefig('q_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

# 保存结果
pd.DataFrame(results).to_csv('nea_hd_fit.csv', index=False)