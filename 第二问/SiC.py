import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

sp10 = Path("附件1.xlsx")  # 一次10°反射透射
sp15 = Path("附件2.xlsx")  # 一次15°反射透射

# 读取文件函数
def read(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    cols = ["波数 (cm-1)","反射率 (%)"]
    df.columns = cols
    out = df[[cols[0], cols[1]]].copy()
    out.columns = ["wn","r"]
    out = out.apply(pd.to_numeric, errors="coerce").dropna()
    out = out.sort_values("wn").reset_index(drop=True)
    return out

# 读取数据并画出原始曲线
sp10_df = read(sp10)
sp15_df = read(sp15)

plt.figure(figsize=(10,4))
plt.plot(sp10_df["wn"], sp10_df["r"], label="10°原始")
plt.plot(sp15_df["wn"], sp15_df["r"], label="15°原始")
plt.xlabel("波数 (cm$^{-1}$)"); plt.ylabel("反射率 (%)"); plt.title("原始反射谱")
plt.legend(); plt.tight_layout(); plt.show()

# 裁切，及进行傅里叶变换前的预处理    [1] ChatGPT, GPT-5, OpenAI, 2025-09-06
# 取出原始数组（方便一起裁剪）
wn10_all = sp10_df["wn"].to_numpy()
r10_all  = sp10_df["r"].to_numpy()
wn15_all = sp15_df["wn"].to_numpy()
r15_all  = sp15_df["r"].to_numpy()
CUTOFF=2000.0
def cut(wn, y_sg, y_raw=None, cutoff=2000.0):
    wn = wn.astype(float)
    mask = (wn >= cutoff)
    wn_cut   = wn[mask]
    y_sg_cut = y_sg[mask]
    y_raw_cut = y_raw[mask] if y_raw is not None else None
    return wn_cut, y_sg_cut, y_raw_cut
wn10_cut, sp10_cut, r10_cut = cut(wn10_all, sp10_df["r"], r10_all)
wn15_cut, sp15_cut, r15_cut = cut(wn15_all, sp15_df["r"], r15_all)

plt.figure(figsize=(10,4))
plt.plot(wn15_cut, sp15_cut, label="15° SG")
plt.plot(wn10_cut, sp10_cut, label="10° SG")
plt.xlabel("波数 (cm$^{-1}$)"); plt.ylabel("反射率 (%)"); plt.title("裁剪后反射谱")
plt.legend(); plt.tight_layout(); plt.show()

# 等间距重采样
def rewrite(wn, y):
    wn = np.asarray(wn, float)
    y = np.asarray(y, float)
    newwn = np.linspace(wn.min(), wn.max(), len(wn))
    newy  = np.interp(newwn, wn, y)
    dsigma = newwn[1] - newwn[0]
    return newwn, newy, dsigma

# 去趋势
def detrend(x, y):
    X = np.vstack([x**k for k in range(4)]).T
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ coef

def prep(wn, y):
    y_d = detrend(wn, y)
    y_d = y_d - np.mean(y_d)
    return y_d

wn10_u, sp10_u, delta10 = rewrite(wn10_cut, sp10_cut)
sp10_p = prep(wn10_u, sp10_u)
wn15_u, sp15_u, delta15 = rewrite(wn15_cut, sp15_cut)
sp15_p = prep(wn15_u, sp15_u)
# 输出图像
plt.figure(figsize=(10,4))
plt.plot(wn10_u, sp10_p, label="10°预处理")
plt.plot(wn15_u, sp15_p, label="15°预处理")
plt.xlabel("波数 (cm$^{-1}$)"); plt.ylabel("反射率 (%)"); plt.title("预处理后反射谱")
plt.legend(); plt.tight_layout(); plt.show()

# 更好的峰值
def betterpeak(F, P, k):
    n = len(P)
    if k <= 0 or k >= n - 1:
        return float(F[k])
    y1, y2, y3 = P[k-1], P[k], P[k+1]
    denom = y1 - 2.0 * y2 + y3
    if abs(denom) < 1e-15:
        return float(F[k])
    delta = 0.5 * (y1 - y3) / denom
    df = F[1] - F[0]
    f = F[k] + delta * df
    return f

# 全谱FFT求全局S(σ), FFT模板由人工智能生成  [1] ChatGPT, GPT-5, OpenAI, 2025-09-07
def globalfft(y, delta, cut=0):
    n = len(y)
    N = int(2**int(np.log2(n)+2.5))
    Y = np.fft.fft(y, n=N)
    F = np.fft.fftfreq(N, d=delta)
    P = np.abs(Y)**2
    m = F >= cut
    F, P = F[m], P[m]
    k = int(np.argmax(P))
    S = betterpeak(F, P, k)
    return S

# 滑窗 FFT求局部S(σ)
def slidefft(wn, y, delta, T=5, forwardT=0.01, cut=5e-4):
    S0= globalfft(y, delta, cut)
    dsigma = 1.0 / S0
    winp = int(T * dsigma / delta)
    sigmas, Ss = [], []
    start = 0
    end = len(wn) - winp + 1
    step = int(forwardT * dsigma / delta)
    for s in range(start, end, step):
        seg = y[s:s+winp]
        N = int(2**int(np.log2(winp)+2.5))
        Y = np.fft.fft(seg, n=N)
        F = np.fft.fftfreq(N, d=delta)
        P = np.abs(Y)**2
        m = F >= cut
        F, P = F[m], P[m]
        k = int(np.argmax(P))
        S= betterpeak(F, P, k)
        Ss.append(S)
        sigmas.append(0.5*(wn[s] + wn[s+winp-1]))
    return np.array(sigmas), np.array(Ss)

def check(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    sigma_hat = 1.4826 * mad
    rel_mad_pct = (sigma_hat / max(abs(med), 1e-12)) * 100.0
    return float(med), float(med - 1.96*sigma_hat), float(med + 1.96*sigma_hat), float(rel_mad_pct)

def answer(s10,s15):
    sin10, sin15 = np.sin(np.deg2rad(10.0)), np.sin(np.deg2rad(15.0))
    n = 2.5
    n2=2.5*2.5
    d10 = s10 / (2 * np.sqrt(n2 - sin10 ** 2))
    d15 = s15 / (2 * np.sqrt(n2 - sin15 ** 2))
    return d10, d15

# 全谱 FFT：给 S 的全局初值
s10 = globalfft(sp10_p, delta10)
s15 = globalfft(sp15_p, delta15)
d10a, d15a = answer(s10,s15)
print(f"[全谱FFT]:")
print(f"S(10°) ≈ {s10.real:.6f} cm, Δσ ≈ {1/s10.real:.2f} cm⁻¹")
print(f"S(15°) ≈ {s15.real:.6f} cm, Δσ ≈ {1/s15.real:.2f} cm⁻¹")
print(f"d(10°) ≈ {d10a*10000:.2f} um, d(15°) ≈ {d15a*10000:.2f} um ")

# 滑窗 FFT：得到 S(σ) 的局部估计

sigmas10, Ss10 = slidefft(wn10_u, sp10_p, delta10)
sigmas15, Ss15 = slidefft(wn15_u, sp15_p, delta15)
left  = max(np.nanmin(sigmas10), np.nanmin(sigmas15))
right = min(np.nanmax(sigmas10), np.nanmax(sigmas15))

# 作图看 S(σ) 的稳定性
plt.figure(figsize=(10,4))
plt.plot(sigmas10, Ss10, '-o', ms=3, label="S10(σ) · 滑窗FFT")
plt.plot(sigmas15, Ss15, '-o', ms=3, label="S15(σ) · 滑窗FFT")
plt.xlabel("波数 σ (cm$^{-1}$)"); plt.ylabel("S(σ) (cm)")
plt.title("局部条纹频率 S(σ)(滑窗FFT)")
plt.legend(); plt.tight_layout(); plt.show()

d10_sigma=Ss10/(2*np.sqrt((0.00005* sigmas10+2.34)**2-np.sin(np.deg2rad(10))**2))
d15_sigma=Ss15/(2*np.sqrt((0.00005* sigmas15+2.34)**2-np.sin(np.deg2rad(15))**2))
m10 = np.isfinite(d10_sigma) & (sigmas10 >= left) & (sigmas10 <= right)
m15 = np.isfinite(d15_sigma) & (sigmas15 >= left) & (sigmas15 <= right)
# 置信检验（稳健统计）
d10_med, d10_lo, d10_hi, rel_d10_mad_pct = check(d10_sigma)
d15_med, d15_lo, d15_hi, rel_d15_mad_pct = check(d15_sigma)
d_med_ref = np.median([d10_med, d15_med])
diff_pct = abs(d10_med - d15_med) / max(abs(d_med_ref), 1e-12) * 100.0

f15 = interp1d(sigmas15[m15], d15_sigma[m15], kind='linear', bounds_error=False, fill_value=np.nan)
x_common = sigmas10[m10]
y10_common = d10_sigma[m10] * 1e4
y15_on_10 = f15(x_common) * 1e4

plt.figure(figsize=(10,4))
plt.plot(x_common, y10_common, '-o', ms=3, label="d(σ) · 10°回代")
plt.plot(x_common, y15_on_10, '-o', ms=3, label="d(σ) · 15°回代（插值到10°网格）", alpha=0.8)
plt.xlabel("波数 σ (cm$^{-1}$)"); plt.ylabel("厚度 d (μm)")
plt.title("厚度 d(σ)（同一 σ 网格对齐）")
plt.legend(); plt.tight_layout(); plt.show()

print("[滑窗FFT]:")
print(f"d(σ)10°: 中位 ≈ {d10_med*10000:.2f} μm, 约95%区间 [{d10_lo*10000:.2f}, {d10_hi*10000:.2f}] μm, 相对MAD ≈ {rel_d10_mad_pct:.2f}%")
print(f"d(σ)15°: 中位 ≈ {d15_med*10000:.2f} μm, 约95%区间 [{d15_lo*10000:.2f}, {d15_hi*10000:.2f}] μm, 相对MAD ≈ {rel_d15_mad_pct:.2f}%")
print(f"厚度一致性（两角中位差异）≈ {diff_pct:.2f}%")