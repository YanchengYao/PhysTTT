import numpy as np
from scipy import signal
import matplotlib.pyplot as plt


def calculate_lf_hf(rr_intervals, fs=4.0):
    """
    计算HRV中的LF和HF成分

    参数:
        rr_intervals: RR间期序列（单位：秒）
        fs: 重采样频率（Hz），默认为4Hz

    返回:
        lf: 低频功率(0.04-0.15Hz)
        hf: 高频功率(0.15-0.4Hz)
        lf_hf_ratio: LF/HF比值
        freqs: 频率数组
        psd: 功率谱密度
    """
    # 将RR间期转换为心率时间序列
    rr_times = np.cumsum(rr_intervals)  # 累积时间点
    rr_values = rr_intervals  # RR间期值

    # 创建均匀时间网格
    time_grid = np.arange(0, rr_times[-1], 1 / fs)

    # 使用线性插值重采样RR间期序列
    interp_func = np.interp(time_grid, rr_times, rr_values)

    # 去趋势（减去均值）
    detrended = interp_func - np.mean(interp_func)

    # 计算功率谱密度(PSD)
    freqs, psd = signal.welch(detrended, fs=fs, nperseg=256)

    # 计算LF和HF功率
    lf_band = (freqs >= 0.04) & (freqs <= 0.15)
    hf_band = (freqs >= 0.15) & (freqs <= 0.4)

    lf = np.trapz(psd[lf_band], freqs[lf_band])
    hf = np.trapz(psd[hf_band], freqs[hf_band])

    lf_hf_ratio = lf / hf

    return lf, hf, lf_hf_ratio, freqs, psd


# 示例使用
if __name__ == "__main__":
    # 生成模拟RR间期数据（单位：秒）
    np.random.seed(42)
    t = np.linspace(0, 300, 300)  # 5分钟数据
    rr_intervals = 0.8 + 0.1 * np.sin(2 * np.pi * 0.1 * t) + 0.05 * np.sin(
        2 * np.pi * 0.25 * t) + 0.02 * np.random.normal(size=len(t))

    # 计算LF和HF
    lf, hf, ratio, freqs, psd = calculate_lf_hf(rr_intervals)

    print(f"LF功率: {lf:.4f}")
    print(f"HF功率: {hf:.4f}")
    print(f"LF/HF比值: {ratio:.4f}")

    # 绘制功率谱
    plt.figure(figsize=(10, 6))
    plt.plot(freqs, psd, label='PSD')
    plt.fill_between(freqs, 0, psd, where=((freqs >= 0.04) & (freqs <= 0.15)), color='red', alpha=0.3,
                     label='LF (0.04-0.15 Hz)')
    plt.fill_between(freqs, 0, psd, where=((freqs >= 0.15) & (freqs <= 0.4)), color='green', alpha=0.3,
                     label='HF (0.15-0.4 Hz)')
    plt.xlim(0, 0.5)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density')
    plt.title('HRV Power Spectrum')
    plt.legend()
    plt.grid()
    plt.show()