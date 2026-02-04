import numpy as np
import pandas as pd
from typing import Tuple, List
from tqdm import tqdm
from scipy.interpolate import interp1d
from scipy.signal import medfilt
from concurrent.futures import ThreadPoolExecutor
from statsmodels.nonparametric.smoothers_lowess import lowess

from utils.log_config import get_logger

logger = get_logger(__name__)


class ECGSignalProcessor:
    """Scale, clean, and process multi-lead ECG signals (spectral scaling, peak detection, lead processing)."""

    def __init__(self, fs=250):
        self.fs = fs

    def compute_magnitude_spectrum(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (fft_freq, magnitude_spectrum) for the given 1D signal."""
        fft_result = np.fft.fft(signal)
        fft_freq = np.fft.fftfreq(len(signal), 1/self.fs)
        magnitude_spectrum = np.abs(fft_result)
        return fft_freq, magnitude_spectrum

    def plot_mean_spectrum(self, signals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        all_magnitude_spectra = []
        signals = signals.astype(np.float32)
        for signal in signals:
            fft_freq, magnitude_spectrum = self.compute_magnitude_spectrum(signal)
            all_magnitude_spectra.append(magnitude_spectrum)
        mean_magnitude_spectrum = np.mean(all_magnitude_spectra, axis=0)
        return fft_freq, mean_magnitude_spectrum
    
    def adjust_spectral_power(self, signal: np.ndarray, power_ratio: float) -> np.ndarray:
        fft_result = np.fft.fft(signal)
        adjusted_fft = fft_result * power_ratio
        adjusted_signal = np.fft.ifft(adjusted_fft)
        return np.real(adjusted_signal)    
    
    def compute_average_spectral_power(self, magnitude_spectrum: np.ndarray) -> float:
        return np.mean(magnitude_spectrum)
    
    def scale_ecg_signals(self, df: pd.DataFrame, power_ratio: float) -> pd.DataFrame:
        """Scale ECG signals in df to match the target power_ratio (e.g. PTBXL); returns df with updated ecg_signal column."""
        _, mean_magnitude_spectrum = self.plot_mean_spectrum(np.array(df['ecg_signal'].tolist())[:, :, 0])
        new_avg_power = self.compute_average_spectral_power(mean_magnitude_spectrum)
        factor = power_ratio / new_avg_power
        
        df['ecg_signal'] = df['ecg_signal'].apply(lambda x: self.adjust_spectral_power(x, factor))
        return df
        
    def detect_peaks_with_sliding_window(
        self, 
        signals: np.ndarray, 
        window_size: int = 3, 
        std_threshold: int = 2, 
        min_freq: int = 20, 
        merge_threshold: int = 1
    ) -> List[Tuple[float, float]]:
        fft_freq, mean_magnitude_spectrum = self.plot_mean_spectrum(signals)
        valid_indices = fft_freq >= min_freq
        valid_freqs = fft_freq[valid_indices]
        valid_magnitude_spectrum = mean_magnitude_spectrum[valid_indices]
        
        freq_resolution = fft_freq[1] - fft_freq[0]
        window_size_points = int(window_size / freq_resolution)
        
        peaks = []
        harmonics = []

        for i in range(len(valid_magnitude_spectrum) - window_size_points + 1):
            window_magnitudes = valid_magnitude_spectrum[i:i+window_size_points]
            mean_magnitude = np.mean(window_magnitudes)
            std_magnitude = np.std(window_magnitudes)

            for j in range(window_size_points):
                current_freq = valid_freqs[i + j]
                current_threshold = std_threshold
                is_harmonic = any(
                    np.isclose(current_freq, fundamental * harmonic_ratio, rtol=0.05)
                    for fundamental in harmonics for harmonic_ratio in [2, 3, 4]
                )
                
                if is_harmonic:
                    current_threshold *= 0.65
                
                if window_magnitudes[j] > mean_magnitude + current_threshold * std_magnitude:
                    peaks.append(i + j)
                    harmonics.append(current_freq)
        
        if not peaks:
            return []
        
        peak_freqs = valid_freqs[peaks]
        
        merged_regions = []
        current_region = [peaks[0]]
        
        for i in range(1, len(peaks)):
            if peak_freqs[i] - peak_freqs[i-1] <= merge_threshold:
                current_region.append(peaks[i])
            else:
                merged_regions.append(current_region)
                current_region = [peaks[i]]
        
        if current_region:
            merged_regions.append(current_region)
        
        region_averages = []
        for region in merged_regions:
            average_freq = np.mean(valid_freqs[region])
            region_averages.append(average_freq)
        
        harmonics_detected = []
        for i, freq1 in enumerate(region_averages):
            for j, freq2 in enumerate(region_averages):
                if i != j and (np.isclose(freq2, freq1 * 0.5, rtol=0.05) or
                               np.isclose(freq2, freq1 * 2, rtol=0.05) or
                               np.isclose(freq2, freq1 * 3, rtol=0.05) or
                               np.isclose(freq2, freq1 * 4, rtol=0.05)):
                    harmonics_detected.append((freq1, freq2))
        
        list_real_range = []
        for region in merged_regions:
            region_start = valid_freqs[region[0]]
            region_end = valid_freqs[region[-1]]
            list_real_range.append((region_start, region_end))
        
        return list_real_range

    def flatten_fft_peak(self, signal: np.ndarray, flatten_ranges: List[Tuple[float, float]] = [(59.5, 60.5), (69.5, 70.5)]) -> np.ndarray:
        if np.isnan(signal).any():
            signal = np.nan_to_num(signal)
        
        fft_result = np.fft.fft(signal)
        freqs = np.fft.fftfreq(len(signal), 1/self.fs)
        n = len(signal) // 2

        fft_phase = np.angle(fft_result)
        fft_magnitude = np.abs(fft_result)
        flatten_ranges = set(flatten_ranges)
        for flatten_range in flatten_ranges:
            if flatten_range[0] >= flatten_range[1]:
                continue

            start_idx = np.searchsorted(freqs[:n], flatten_range[0], side='left')
            end_idx = np.searchsorted(freqs[:n], flatten_range[1], side='right') - 1

            start_idx = max(start_idx, 0)
            end_idx = min(end_idx, n - 1)

            if start_idx != end_idx:
                interp_mag = interp1d(
                    [freqs[start_idx], freqs[end_idx]], 
                    [fft_magnitude[start_idx], fft_magnitude[end_idx]], 
                    kind='linear', fill_value="extrapolate"
                )
                
                indices = np.arange(start_idx, end_idx + 1)
                interpolated_magnitudes = interp_mag(freqs[indices])

                fft_result[indices] = interpolated_magnitudes * np.exp(1j * fft_phase[indices])
                
                if start_idx > 0:
                    fft_result[-indices] = interpolated_magnitudes * np.exp(1j * fft_phase[-indices])

        modified_signal = np.fft.ifft(fft_result)
        return modified_signal.real

    def process_signals_parallel(
        self, 
        signals: np.ndarray, 
        flatten_ranges: List[Tuple[float, float]] = [(59.5, 60.5), (69.5, 70.5)], 
        max_workers: int = 4
    ) -> np.ndarray:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.flatten_fft_peak, signal, flatten_ranges) for signal in signals]
            results = [future.result() for future in futures]
        return np.array(results)

    def plot_mean_spectrum_with_median_loess(
        self, 
        signals: np.ndarray, 
        color: str = 'blue', 
        label: str = 'MHI', 
        frac: float = 0.10, 
        kernel_size: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        fft_freq, mean_magnitude_spectrum = self.plot_mean_spectrum(signals, color=color, label=label)
        
        valid_indices = fft_freq >= 40
        valid_freqs = fft_freq[valid_indices]
        valid_magnitude_spectrum = mean_magnitude_spectrum[valid_indices]

        median_filtered = medfilt(valid_magnitude_spectrum, kernel_size=kernel_size)
        loess_smoothed = lowess(median_filtered, valid_freqs, frac=frac, return_sorted=False)

        return valid_freqs, loess_smoothed

    def find_crossings_for_peaks(
        self, 
        signals: np.ndarray, 
        peak_ranges: List[Tuple[float, float]], 
        frac: float = 0.1, 
        kernel_size: int = 5
    ) -> List[Tuple[float, float]]:
        fft_freq, mean_magnitude_spectrum = self.plot_mean_spectrum(signals)
        
        valid_indices = fft_freq >= 30
        valid_freqs = fft_freq[valid_indices]
        valid_magnitude_spectrum = mean_magnitude_spectrum[valid_indices]

        median_filtered = medfilt(valid_magnitude_spectrum, kernel_size=kernel_size)
        loess_smoothed = lowess(median_filtered, valid_freqs, frac=frac, return_sorted=False)

        crossings = []
        for peak_start, peak_end in peak_ranges:
            midpoint = (peak_start + peak_end) / 2
            closest_to_mid = np.argmin(np.abs(valid_freqs - midpoint))
            
            upstream_crossing = None
            for i in range(closest_to_mid, 0, -1):
                if (valid_magnitude_spectrum[i-1] > loess_smoothed[i-1] and valid_magnitude_spectrum[i] < loess_smoothed[i]) or \
                   (valid_magnitude_spectrum[i-1] < loess_smoothed[i-1] and valid_magnitude_spectrum[i] > loess_smoothed[i]):
                    upstream_crossing = valid_freqs[i]
                    break

            downstream_crossing = None
            for i in range(closest_to_mid, len(valid_freqs) - 1):
                if (valid_magnitude_spectrum[i] > loess_smoothed[i] and valid_magnitude_spectrum[i+1] < loess_smoothed[i+1]) or \
                   (valid_magnitude_spectrum[i] < loess_smoothed[i] and valid_magnitude_spectrum[i+1] > loess_smoothed[i+1]):
                    downstream_crossing = valid_freqs[i]
                    break

            if upstream_crossing is not None and downstream_crossing is not None:
                crossings.append((upstream_crossing, downstream_crossing))
            elif upstream_crossing is None and downstream_crossing is not None:
                crossings.append((peak_start, downstream_crossing))
            elif upstream_crossing is not None and downstream_crossing is None:
                crossings.append((upstream_crossing, peak_end))
            else:
                crossings.append((peak_start, peak_end))

        return crossings

    def widen_ranges(self, ranges_list: List[Tuple[float, float]], widen_by: int = 1) -> List[Tuple[float, float]]:
        return [(start - widen_by, end + widen_by) for start, end in ranges_list]

    def clean_and_process_ecg_leads(
        self,
        df: pd.DataFrame,
        window_size: int = 5,
        std_threshold: int = 5,
        max_workers: int = 16,
    ) -> pd.DataFrame:
        """Detect peaks and clean each lead; returns df with processed ecg_signal column."""
        logger.info("Step 1: Detecting peaks")
        input_data = np.array(df['ecg_signal'].tolist())
        logger.debug("Input shape: %s", input_data.shape)
        peak_ranges = self.detect_peaks_with_sliding_window(
            input_data[:, :, 0],
            window_size=window_size,
            std_threshold=std_threshold
        )
        
        if len(peak_ranges) == 0:
            logger.warning("No peaks detected; signal similar to flat/MHI, returning unchanged.")
            return df
        processed_leads = []
        logger.info("Step 2: Processing leads (this step can be slow)")
        for lead_index in tqdm(range(12)):
            logger.debug("Processing lead %d", lead_index)
            crossings = self.find_crossings_for_peaks(input_data[:, :, lead_index], peak_ranges=peak_ranges)
            widened_crossings = self.widen_ranges(crossings)

            cleaned_lead = self.process_signals_parallel(input_data[:, :, lead_index].astype(np.float32), flatten_ranges=widened_crossings, max_workers=max_workers)
            processed_leads.append(cleaned_lead.astype(np.float32))

        cleaned_data = np.array(processed_leads).astype(np.float32)
        cleaned_data = np.swapaxes(cleaned_data, 0, -2)
        cleaned_data = np.swapaxes(cleaned_data, -1, -2)

        return pd.DataFrame({
            'ecg_path': df['ecg_path'].tolist(),
            'ecg_signal': list(cleaned_data)
        })

if __name__ == "__main__":
    root_dir = "./tmp/"
    
    ecg_signal_processor = ECGSignalProcessor()
    ecg_signal_processor.process_batch(root_dir, num_workers=4)        

        

        
