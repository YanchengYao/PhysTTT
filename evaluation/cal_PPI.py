import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import find_peaks


def calculate_ppi_from_ppg(ppg_signal, fs, plot=False):
    """
    从PPG信号计算PPI(脉搏波间期)序列

    参数:
        ppg_signal: PPG信号数据
        fs: 采样频率(Hz)
        plot: 是否绘制检测结果

    返回:
        ppi_intervals: PPI间隔序列(秒)
        peak_indices: 脉搏波峰值位置的索引数组
    """
    # 1. 信号预处理
    # 带通滤波(0.5-5Hz)去除噪声和基线漂移
    b, a = signal.butter(4, [0.5, 5], btype='bandpass', fs=fs)
    filtered_ppg = signal.filtfilt(b, a, ppg_signal)

    # 2. 峰值检测
    # 计算移动平均作为动态阈值
    window_size = int(fs * 0.6)  # 600ms窗口
    moving_avg = np.convolve(filtered_ppg, np.ones(window_size) / window_size, mode='same')

    # 寻找高于移动平均1.2倍的峰值
    height_threshold = 1.2 * moving_avg
    peaks, properties = find_peaks(filtered_ppg,
                                   height=height_threshold,
                                   distance=int(fs * 0.6))  # 最小间隔0.6秒

    # 3. 计算PPI间期(秒)
    peak_times = peaks / fs
    ppi_intervals = np.diff(peak_times)

    # 4. 异常值处理 (使用中位数绝对偏差)
    median_ppi = np.median(ppi_intervals)
    mad = 1.4826 * np.median(np.abs(ppi_intervals - median_ppi))  # 1.4826是高斯分布的比例因子
    valid_mask = np.abs(ppi_intervals - median_ppi) < 3 * mad  # 3倍MAD阈值
    cleaned_ppi = ppi_intervals[valid_mask]

    if plot:
        plt.figure(figsize=(12, 6))
        plt.plot(np.arange(len(ppg_signal)) / fs, ppg_signal, label='原始PPG')
        plt.plot(np.arange(len(filtered_ppg)) / fs, filtered_ppg, label='滤波后PPG')
        plt.plot(peaks / fs, filtered_ppg[peaks], 'rx', label='检测到的峰值')
        plt.xlabel('时间 (秒)')
        plt.ylabel('幅值')
        plt.title('PPG信号峰值检测')
        plt.legend()
        plt.grid()
        plt.show()

        plt.figure(figsize=(12, 4))
        plt.plot(peak_times[:-1], ppi_intervals * 1000, 'bo-', label='原始PPI')
        plt.plot(peak_times[:-1][valid_mask], cleaned_ppi * 1000, 'go', label='有效PPI')
        plt.xlabel('时间 (秒)')
        plt.ylabel('PPI间期 (ms)')
        plt.title('PPI间期序列')
        plt.legend()
        plt.grid()
        plt.show()

    return cleaned_ppi, peaks[valid_mask]


# 示例使用
if __name__ == "__main__":
    # 生成模拟PPG信号
    fs = 100  # 采样频率100Hz
    duration = 30  # 30秒数据
    t = np.linspace(0, duration, int(fs * duration))

    # 基础心率(随时间变化)
    base_hr = 60 + 5 * np.sin(2 * np.pi * 0.1 * t)  # 心率在55-65bpm之间变化

    # 生成PPG信号 (模拟脉搏波)
    ppg_signal = np.zeros_like(t)
    for i in range(len(t)):
        # 每个心跳生成一个脉搏波
        if i == 0 or (i > 0 and t[i] - t[i - 1] > 0.6):  # 至少间隔600ms
            pulse_width = 0.3  # 脉搏波宽度
            pulse = np.exp(-40 * (t - t[i]) ** 2)  # 高斯波形模拟脉搏
            ppg_signal += pulse * (0.8 + 0.2 * np.random.rand())  # 添加随机幅值变化

    # 添加噪声
    ppg_signal += 0.1 * np.random.normal(size=len(t))

    # 计算PPI间期
    ppi_intervals, peak_indices = calculate_ppi_from_ppg(ppg_signal, fs, plot=True)

    # 打印结果
    print(f"检测到{len(peak_indices)}个脉搏波峰值")
    print(f"平均心率: {60 / np.mean(ppi_intervals):.1f} bpm")
    print("前10个PPI间期(ms):", np.round(ppi_intervals[:10] * 1000, 1))