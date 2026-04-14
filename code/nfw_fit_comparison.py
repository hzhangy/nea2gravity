import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import warnings

# 模型函数（包含气体、盘、核球和 NFW 晕）
def exponential_disk(R, Vd0, Rd):
    """指数盘旋转速度贡献"""
    x = R / Rd
    return Vd0 * np.sqrt(1 - (1 + x) * np.exp(-x))

def NFW_halo(R, Vmax, Rmax, c=10):
    """NFW 暗物质晕，固定浓度 c=10"""
    x = R / Rmax
    f = np.log(1 + c * x) - (c * x) / (1 + c * x)
    f200 = np.log(1 + c) - c / (1 + c)
    return Vmax * np.sqrt(f / f200)

def total_velocity(R, Vd0, Rd, Vmax, Rmax, Vgas, Vdisk, Vbulge):
    """总速度：气体 + 盘 + 核球 + 暗物质晕"""
    Vd = exponential_disk(R, Vd0, Rd)
    Vh = NFW_halo(R, Vmax, Rmax)
    return np.sqrt(Vgas**2 + Vdisk**2 + Vbulge**2 + Vd**2 + Vh**2)

# 读取数据
col_names = ['Galaxy', 'R', 'Vobs', 'Verr', 'Vgas', 'Vdisk', 'Vbulge', 'Vhalo', 'Flag']
df = pd.read_csv('table2.txt', sep=r'\s+', names=col_names, comment=None, header=None)

# 处理速度误差：将 0 替换为一个小值（如 0.1 或 1% 观测速度）
def fix_verr(row):
    if row['Verr'] == 0:
        return max(0.1, 0.01 * row['Vobs'])
    return row['Verr']

df['Verr'] = df.apply(fix_verr, axis=1)

results = []

for gal in df['Galaxy'].unique():
    g = df[df['Galaxy'] == gal].sort_values('R')
    R = g['R'].values
    Vobs = g['Vobs'].values
    Verr = g['Verr'].values
    Vgas = g['Vgas'].values
    Vdisk = g['Vdisk'].values
    Vbulge = g['Vbulge'].values

    # 数据点太少则跳过
    if len(R) < 4:
        print(f"{gal}: 数据点不足 ({len(R)} 个)，跳过")
        continue

    # 初始参数估计
    # 盘参数：如果 Vdisk 非零，用其最大值作为 Vd0 初值，否则用观测最大速度的 30%
    if np.max(Vdisk) > 0:
        Vd0_guess = np.max(Vdisk)
    else:
        Vd0_guess = 0.3 * np.max(Vobs)
    Rd_guess = np.median(R)  # 盘尺度半径

    # 暗物质晕参数：先用观测速度减去气体、盘、核球贡献的平方和的平方根
    Vother = np.sqrt(Vgas**2 + Vdisk**2 + Vbulge**2)
    Vhalo_obs = np.sqrt(np.maximum(Vobs**2 - Vother**2, 0))
    Vmax_guess = np.max(Vhalo_obs) if np.max(Vhalo_obs) > 0 else 0.5 * np.max(Vobs)
    Rmax_guess = np.max(R) * 0.5

    p0 = [Vd0_guess, Rd_guess, Vmax_guess, Rmax_guess]
    bounds = (
        [0, 0.01, 0, 0.01],   # 下界
        [np.inf, np.inf, np.inf, np.inf]   # 上界
    )

    # 包装函数，将气体、盘、核球作为额外参数传入
    def model(R, Vd0, Rd, Vmax, Rmax):
        return total_velocity(R, Vd0, Rd, Vmax, Rmax, Vgas, Vdisk, Vbulge)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # 忽略内部警告
            popt, pcov = curve_fit(model, R, Vobs, sigma=Verr, p0=p0, bounds=bounds, maxfev=5000)

        # 计算拟合优度
        Vfit = model(R, *popt)
        residuals = Vobs - Vfit
        chi2 = np.sum((residuals / Verr)**2)
        ndof = len(R) - len(popt)
        chi2_red = chi2 / ndof if ndof > 0 else np.inf

        # 检查协方差矩阵是否奇异
        if np.isinf(pcov).any() or np.isnan(pcov).any():
            cov_warning = True
        else:
            cov_warning = False

        results.append({
            'Galaxy': gal,
            'Vd0': popt[0],
            'Rd': popt[1],
            'Vmax': popt[2],
            'Rmax': popt[3],
            'chi2_red': chi2_red,
            'cov_warning': cov_warning,
            'ndata': len(R),
            'fit_success': True
        })
        print(f"{gal}: 拟合完成，χ²/ν = {chi2_red:.2f}" + (" (协方差奇异)" if cov_warning else ""))
    except Exception as e:
        print(f"{gal}: 拟合失败 - {e}")
        results.append({'Galaxy': gal, 'fit_success': False, 'error': str(e)})

# 输出统计
success = [r for r in results if r.get('fit_success', False)]
good = [r for r in success if r.get('chi2_red', np.inf) < 5 and not r.get('cov_warning', True)]
print(f"\n成功拟合: {len(success)} 个星系")
print(f"其中拟合质量良好（χ²/ν < 5 且协方差正常）: {len(good)} 个")