import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
col_names = ['Galaxy', 'R', 'Vobs', 'Verr', 'Vgas', 'Vdisk', 'Vbulge', 'Vhalo', 'Flag']
df = pd.read_csv('table2.txt', sep=r'\s+', names=col_names)
df['Galaxy'] = df['Galaxy'].astype(str).str.strip()
g = df[df['Galaxy'] == '3.91'].sort_values('R')

R = g['R'].values
Vobs = g['Vobs'].values
Verr = g['Verr'].values
Vbar = np.sqrt(g['Vgas']**2 + g['Vdisk']**2 + g['Vbulge']**2).values

# 拟合参数
q = 1.154
Rc = 3.16

# 模型计算
V_model = Vbar * (1.0 + R / Rc) ** ((2.0 - q) / 2.0)

# 按 R 排序（数据已排序，但为绘图保险）
idx = np.argsort(R)
R_sorted = R[idx]
V_model_sorted = V_model[idx]

# 绘图
plt.figure(figsize=(8,5))
plt.errorbar(R, Vobs, yerr=Verr, fmt='ko', ecolor='gray', capsize=2, label='Observed (SPARC)')
plt.plot(R_sorted, V_model_sorted, 'r-', lw=2, label=f'N.E.A. fit (q={q:.3f}, Rc={Rc:.2f} kpc)')

plt.xlabel('Radius (kpc)')
plt.ylabel('Velocity (km/s)')
plt.title('NGC 3198 Rotation Curve')
plt.legend()
plt.grid(alpha=0.3)

# 自动调整坐标轴，留出 5% 边距
plt.xlim(0, R.max() * 1.05)
all_velocities = np.concatenate([Vobs, V_model_sorted])
plt.ylim(0, all_velocities.max() * 1.05)

plt.tight_layout()
plt.savefig('NGC3198_fit_nea.png', dpi=150)
plt.show()

# 同时打印模型值与观测值对比，供您检查
print("R(kpc)  Vobs  Vbar  Vmodel  Residual")
for i in range(min(10, len(R))):
    print(f"{R[i]:6.2f} {Vobs[i]:6.1f} {Vbar[i]:6.1f} {V_model[i]:6.1f} {Vobs[i]-V_model[i]:7.1f}")