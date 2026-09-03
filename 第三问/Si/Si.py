import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import interp1d
from scipy.signal import find_peaks

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

sp10 = Path("附件3.xlsx")  # 一次10°反射透射
sp15 = Path("附件4.xlsx")  # 一次15°反射透射
n=3.47  # 硅折射率（近似常数）

# 读取文件函数
def read(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    cols = ["波数 (cm-1)","反射率 (%)"]
    df.columns = cols
    out = df[[cols[0], cols[1]]].copy()
    out.columns = ["wn","rx"]
    out = out.apply(pd.to_numeric, errors="coerce").dropna()
    out = out.sort_values("wn").reset_index(drop=True)
    return out

# 读取数据并画出原始曲线
sp10_df = read(sp10)
sp15_df = read(sp15)
sp10_df["r"] = sp10_df["rx"] / 100.0
sp15_df["r"] = sp15_df["rx"] / 100.0
plt.figure(figsize=(10,4))
plt.plot(sp10_df["wn"], sp10_df["r"], label="10°原始")
plt.plot(sp15_df["wn"], sp15_df["r"], label="15°原始")
plt.xlabel("波数 (cm$^{-1}$)"); plt.ylabel("反射率"); plt.title("原始反射谱")
plt.legend(); plt.tight_layout(); plt.show()

# 裁切
wn10_all = sp10_df["wn"].to_numpy()
r10_all  = sp10_df["r"].to_numpy()
wn15_all = sp15_df["wn"].to_numpy()
r15_all  = sp15_df["r"].to_numpy()
CUTOFF=400.0
def cut(wn, y_sg, y_raw=None, cutoff=400.0):
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
plt.xlabel("波数 (cm$^{-1}$)"); plt.ylabel("反射率"); plt.title("裁剪后反射谱")
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

# 处理 10° 曲线
wn10_u, sp10_u, delta10 = rewrite(wn10_cut, sp10_cut)
sp10_p = prep(wn10_u, sp10_u)
# 处理 15° 曲线
wn15_u, sp15_u, delta15 = rewrite(wn15_cut, sp15_cut)
sp15_p = prep(wn15_u, sp15_u)
# 输出图像
plt.figure(figsize=(10,4))
plt.plot(wn10_u, sp10_p, label="10°预处理")
plt.plot(wn15_u, sp15_p, label="15°预处理")
plt.xlabel("波数 (cm$^{-1}$)"); plt.ylabel("反射率"); plt.title("预处理后反射谱")
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

# 全谱FFT求全局S(σ)
def globalfft(y, delta, cut=0, ret=False):
    n = len(y)
    N = int(2**int(np.log2(n)+2.5))
    Y = np.fft.fft(y, n=N)
    F = np.fft.fftfreq(N, d=delta)
    P = np.abs(Y)**2
    m = F >= cut
    F, P = F[m], P[m]
    k = int(np.argmax(P))
    S = float(betterpeak(F, P, k))
    if ret:
        return S, F, P, k
    return S

# 滑窗 FFT求局部S(σ)
def slidefft(wn, y, delta, T=7, forwardT=0.01, cut=5e-4, return_spec=False):
    S0, _, _, _ = globalfft(y, delta, cut, ret=True)
    dsigma = 1.0 / S0
    winp = int(T * dsigma / delta)
    if winp < 3:
        if return_spec:
            return np.array([]), np.array([]), np.array([]), np.empty((0, 0))
        else:
            return np.array([]), np.array([])

    step = max(1, int(forwardT * dsigma / delta))
    start = 0
    end = len(wn) - winp + 1

    sigmas, Ss = [], []
    P_rows = []
    F_ref = None

    for s in range(start, end, step):
        seg = y[s:s+winp]
        N = int(2**int(np.log2(winp)+2.5))
        Y = np.fft.fft(seg, n=N)
        F = np.fft.fftfreq(N, d=delta)
        P = np.abs(Y)**2
        m = F >= cut
        F, P = F[m], P[m]
        if F_ref is None:
            F_ref = F
        else:
            if len(F) != len(F_ref) or np.max(np.abs(F - F_ref)) > 1e-12:
                P = np.interp(F_ref, F, P, left=0.0, right=0.0)
                F = F_ref
        k = int(np.argmax(P))
        S = betterpeak(F, P, k)
        Ss.append(S)
        sigmas.append(0.5 * (wn[s] + wn[s+winp-1]))
        if return_spec:
            P_rows.append(P)
    sigmas = np.array(sigmas)
    Ss = np.array(Ss)
    if return_spec:
        P_mat = np.vstack(P_rows) if P_rows else np.empty((0, 0))
        return sigmas, Ss, F_ref, P_mat
    return sigmas, Ss

def answer(s10,s15):
    sin10, sin15 = np.sin(np.deg2rad(10.0)), np.sin(np.deg2rad(15.0))
    n2 = 3.47**2
    n = 3.47
    d10 = s10 / (2 * np.sqrt(n2 - sin10 ** 2))
    d15 = s15 / (2 * np.sqrt(n2 - sin15 ** 2))
    return n, d10, d15

def plot_global_spectrum(F, P, title=""):
    plt.figure(figsize=(10, 4))
    plt.plot(F, P, lw=1)
    y = find_peaks(P, height=8000)
    plt.plot(F[y[0]], y[1]['peak_heights'], 'x')
    plt.xlim(0, 0.015)
    plt.xlabel("S (cm)")
    plt.ylabel("|FFT|$^2$")
    plt.title(title)
    plt.tight_layout(); plt.show()
    print(f"检测到的峰值频率S: {F[y[0]][0]:.6f}(σ), {F[y[0]][1]:.6f}({F[y[0]][1]/F[y[0]][0]:.2f}σ)")

def plot_spectrogram(sigmas, F, P_mat, Ss=None, title=""):
    if P_mat.size == 0:
        return
    plt.figure(figsize=(10, 4))
    extent = [sigmas.min(), sigmas.max(), F.min(), F.max()]
    plt.imshow(np.log1p(P_mat).T, extent=extent, origin="lower", aspect="auto")
    plt.ylim(0, 0.015)
    if Ss is not None and len(Ss) == len(sigmas):
        plt.plot(sigmas, Ss, 'w-', lw=1, label="峰值 S(σ)")
        plt.legend(loc="upper right")
    plt.xlabel("波数 σ (cm$^{-1}$)")
    plt.ylabel("S (cm)")
    plt.title(title + "（对数功率）")
    plt.colorbar(label="log(1+|FFT|$^2$)")
    plt.tight_layout(); plt.show()
print(f"代入已知数据:硅晶体在400到4000波数下n = 3.47 （近似常数）")
# 全谱 FFT：给 S 的全局初值
s10, F10, P10, _ = globalfft(sp10_p, delta10, cut=5e-4, ret=True)
s15, F15, P15, _ = globalfft(sp15_p, delta15, cut=5e-4, ret=True)

# 画全局频谱（横轴是 S，纵轴是 |FFT|^2）
plot_global_spectrum(F10, P10, "10° 全谱FFT功率谱")
plot_global_spectrum(F15, P15, "15° 全谱FFT功率谱")

n, d10a, d15a = answer(s10,s15)
print(f"[全谱FFT]:")
print(f"S(10°) ≈ {s10.real:.6f} cm, Δσ ≈ {1/s10.real:.2f} cm⁻¹")
print(f"S(15°) ≈ {s15.real:.6f} cm, Δσ ≈ {1/s15.real:.2f} cm⁻¹")
print(f"d(10°) ≈ {d10a*10000:.2f} um, d(15°) ≈ {d15a*10000:.2f} um ")
diff=abs(d10a - d15a)
diff_pct = diff / ((d10a + d15a)/2) * 100
print(f"厚度一致性（两角中位差异）≈ {diff_pct:.2f}%")