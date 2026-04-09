#!/usr/bin/env python3
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def parse_table1(filename):
    """解析 Table1.mrt，返回星系名到参数的字典"""
    gal_data = {}
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    start = 0
    for i, line in enumerate(lines):
        if line.strip().startswith(('Name', 'Galaxy')):
            start = i + 1
            break
    for line in lines[start:]:
        line = line.strip()
        if not line or line.startswith('-'):
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

def load_rotation_curves(filename):
    """手动解析 table2.txt，避免 pandas 的列错位问题"""
    data = []  # 每个元素: (galaxy, R, Vobs, Verr, Vgas, Vdisk, Vbulge)
    with open(filename, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            # 按空白分割
            parts = line.split()
            if len(parts) < 9:
                print(f"警告: 第 {line_num} 行字段数不足 ({len(parts)}), 跳过")
                continue
            galaxy = parts[0]
            # 检查星系名是否以字母开头（避免将数字作为星系名）
            if not galaxy[0].isalpha():
                # 可能是数值，跳过这一行（但实际 CamB 是字母，这里应该不会触发）
                continue
            try:
                r = float(parts[1])
                vobs = float(parts[2])
                verr = float(parts[3])
                vgas = float(parts[4])
                vdisk = float(parts[5])
                vbulge = float(parts[6])
                # 忽略 Vhalo 和 Flag
                data.append([galaxy, r, vobs, verr, vgas, vdisk, vbulge])
            except ValueError as e:
                print(f"警告: 第 {line_num} 行数值转换失败: {e}")
                continue
    # 转换为 numpy 结构以便分组
    return data

def nea_model(R, q, Rc, Vbar):
    term = (R / Rc) ** (2 - q)
    return Vbar * np.sqrt(1 + term)

def fit_galaxy_nea(R, Vobs, Verr, Vgas, Vdisk, Vbulge):
    Vbar = np.sqrt(Vgas**2 + Vdisk**2 + Vbulge**2)
    mask = (Vbar > 0) & (R > 0)
    if np.sum(mask) < 4:
        return None, None, None, None, False
    R_fit = R[mask]
    Vobs_fit = Vobs[mask]
    Verr_fit = Verr[mask]
    Vbar_fit = Vbar[mask]
    def func(R, q, Rc):
        return Vbar_fit * np.sqrt(1 + (R / Rc) ** (2 - q))
    q_guess = 1.0
    Rc_guess = np.median(R_fit)
    try:
        popt, pcov = curve_fit(func, R_fit, Vobs_fit, sigma=Verr_fit,
                               p0=[q_guess, Rc_guess],
                               bounds=([1.0, 0.01], [2.0, 500.0]),
                               maxfev=5000)
        q, Rc = popt
        Vpred = func(R_fit, q, Rc)
        residuals = Vobs_fit - Vpred
        chi2 = np.sum((residuals / Verr_fit)**2)
        ndof = len(R_fit) - 2
        chi2_red = chi2 / ndof if ndof > 0 else np.inf
        cov_ok = not (np.isinf(pcov).any() or np.isnan(pcov).any())
        return q, Rc, chi2_red, cov_ok, True
    except Exception as e:
        return None, None, None, None, False

def main():
    print("解析 Table1.mrt...")
    t1_data = parse_table1('Table1.mrt')
    print(f"Table1 星系数: {len(t1_data)}")
    
    print("加载 table2.txt...")
    raw_data = load_rotation_curves('table2.txt')
    print(f"table2 总数据行数: {len(raw_data)}")
    
    # 按星系名分组
    from collections import defaultdict
    groups = defaultdict(list)
    for entry in raw_data:
        gal = entry[0]
        groups[gal].append(entry[1:])  # 存储 R, Vobs, Verr, Vgas, Vdisk, Vbulge
    
    print(f"table2 中星系数: {len(groups)}")
    t1_names = set(t1_data.keys())
    matched_names = [name for name in groups.keys() if name in t1_names]
    print(f"匹配星系数: {len(matched_names)}")
    if len(matched_names) == 0:
        print("匹配失败，示例: table2 中前几个 =", list(groups.keys())[:10])
        print("Table1 中前几个 =", list(t1_names)[:10])
        return
    
    results = []
    for gal in matched_names:
        entries = groups[gal]
        # 转换为 numpy 数组
        arr = np.array(entries)
        R = arr[:,0].astype(float)
        Vobs = arr[:,1].astype(float)
        Verr = arr[:,2].astype(float)
        Vgas = arr[:,3].astype(float)
        Vdisk = arr[:,4].astype(float)
        Vbulge = arr[:,5].astype(float)
        # 排序
        idx = np.argsort(R)
        R = R[idx]
        Vobs = Vobs[idx]
        Verr = Verr[idx]
        Vgas = Vgas[idx]
        Vdisk = Vdisk[idx]
        Vbulge = Vbulge[idx]
        
        q, Rc, chi2_red, cov_ok, success = fit_galaxy_nea(R, Vobs, Verr, Vgas, Vdisk, Vbulge)
        if success:
            results.append({
                'Galaxy': gal,
                'q': q,
                'Rc': Rc,
                'chi2_red': chi2_red,
                'cov_ok': cov_ok,
                'ndata': len(R)
            })
            print(f"{gal}: q={q:.3f}, Rc={Rc:.2f}, χ²/ν={chi2_red:.2f}")
        else:
            print(f"{gal}: 拟合失败")
    
    import pandas as pd
    df_fit = pd.DataFrame(results)
    df_fit.to_csv('nea_hd_fit_corrected.csv', index=False)
    print(f"\n拟合成功 {len(df_fit)} 个星系")
    
    good = df_fit[(df_fit['chi2_red'] < 5) & (df_fit['cov_ok'] == True)]
    print(f"高质量星系: {len(good)}")
    if len(good) == 0:
        return
    
    # 合并 Table1 数据
    t1_df = pd.DataFrame(t1_data).T.reset_index().rename(columns={'index': 'Galaxy'})
    merged = good.merge(t1_df, on='Galaxy', how='inner')
    merged['Mstar'] = merged['L36'] * 0.5
    merged['Mgas'] = merged['MHI'] * 1.33
    merged['Mbar'] = merged['Mstar'] + merged['Mgas']
    merged['Sigma_bar'] = merged['Mbar'] / (np.pi * merged['Reff']**2)
    
    # 重新读取原始数据用于不变量计算（需要所有数据点）
    all_data = []
    for gal in good['Galaxy']:
        entries = groups[gal]
        arr = np.array(entries)
        R = arr[:,0].astype(float)
        Vobs = arr[:,1].astype(float)
        Vbar = np.sqrt(arr[:,3].astype(float)**2 + arr[:,4].astype(float)**2 + arr[:,5].astype(float)**2)
        all_data.append((gal, R, Vobs, Vbar))
    
    all_I = []
    stats = []
    for _, row in merged.iterrows():
        gal = row['Galaxy']
        q = row['q']
        Rc = row['Rc']
        # 找到该星系的数据
        for (g, R, Vobs, Vbar) in all_data:
            if g == gal:
                break
        else:
            continue
        safe_Vbar = np.maximum(Vbar, 1e-5)
        x = R / Rc
        term = x ** (2 - q)
        I = (Vobs / safe_Vbar)**2 - term
        valid = (x > 0.05) & (x < 20) & (Vobs/safe_Vbar > 0) & (Vobs/safe_Vbar < 3) & np.isfinite(I)
        if not any(valid):
            continue
        all_I.extend(I[valid])
        stats.append({
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
    
    if len(all_I) == 0:
        print("无有效不变量")
        return
    
    mean_I = np.mean(all_I)
    std_I = np.std(all_I)
    print("\n" + "="*50)
    print("全息不变量审计结果")
    print("="*50)
    print(f"参与星系: {len(stats)}")
    print(f"I 均值: {mean_I:.4f} (理论值 1)")
    print(f"I 标准差: {std_I:.4f}")
    
    plt.figure(figsize=(8,6))
    plt.hist(all_I, bins=60, range=(0.5, 1.5), color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(1.0, color='red', linestyle='--', lw=2, label='Theoretical I=1')
    plt.xlabel('Holographic invariant I')
    plt.ylabel('Count')
    plt.title(f'Final Audit: mean={mean_I:.3f}, std={std_I:.3f}')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.savefig('final_holo_audit.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("图片已保存为 final_holo_audit.png")
    
    pd.DataFrame(stats).to_csv('nea_precise_audit.csv', index=False)
    print("详细统计保存至 nea_precise_audit.csv")

if __name__ == "__main__":
    main()