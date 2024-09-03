import numpy as np
from tqdm import tqdm
from scipy.interpolate import interp1d
from scipy.signal import medfilt
from concurrent.futures import ThreadPoolExecutor
from statsmodels.nonparametric.smoothers_lowess import lowess

class ECGSignalProcessor:
    def __init__(self, fs=250):
        self.fs = fs

    def compute_magnitude_spectrum(self, signal):
        fft_result = np.fft.rfft(signal)
        fft_freq = np.fft.rfftfreq(len(signal), 1/self.fs)
        magnitude_spectrum = np.abs(fft_result)
        return fft_freq, magnitude_spectrum

    def plot_mean_spectrum(self, signals):
        all_magnitude_spectra = [self.compute_magnitude_spectrum(signal)[1] for signal in signals]
        mean_magnitude_spectrum = np.mean(all_magnitude_spectra, axis=0)
        fft_freq = np.fft.rfftfreq(len(signals[0]), 1/self.fs)
        return fft_freq, mean_magnitude_spectrum

    def detect_peaks_with_sliding_window(self, signals, window_size=3, std_threshold=2, min_freq=20, merge_threshold=1):
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
            
            current_freqs = valid_freqs[i:i+window_size_points]
            is_harmonic = np.any([np.isclose(current_freqs, fundamental * harmonic_ratio, rtol=0.05)
                                  for fundamental in harmonics for harmonic_ratio in [2, 3, 4]], axis=0)
            
            current_threshold = np.where(is_harmonic, std_threshold * 0.65, std_threshold)
            
            peak_indices = np.where(window_magnitudes > mean_magnitude + current_threshold * std_magnitude)[0]
            peaks.extend(i + peak_indices)
            harmonics.extend(current_freqs[peak_indices])
        
        peak_freqs = valid_freqs[peaks]
        peak_magnitudes = valid_magnitude_spectrum[peaks]
        
        # Merge regions
        merged_regions = np.split(peaks, np.where(np.diff(peak_freqs) > merge_threshold)[0] + 1)
        region_averages = [np.mean(valid_freqs[region]) for region in merged_regions]
        
        # Detect harmonics
        harmonics_detected = [(freq1, freq2) for i, freq1 in enumerate(region_averages)
                              for j, freq2 in enumerate(region_averages) if i != j
                              and any(np.isclose(freq2, freq1 * ratio, rtol=0.05) for ratio in [0.5, 2, 3, 4])]
        
        list_real_range = [(valid_freqs[region[0]], valid_freqs[region[-1]]) for region in merged_regions]
        
        return peaks, peak_freqs, peak_magnitudes, merged_regions, region_averages, harmonics_detected, list_real_range

    def flatten_fft_peak(self, signal, flatten_ranges=[(59.5, 60.5), (69.5, 70.5)]):
        signal = np.nan_to_num(signal)
        
        fft_result = np.fft.rfft(signal)
        freqs = np.fft.rfftfreq(len(signal), 1/self.fs)
        
        fft_phase = np.angle(fft_result)
        fft_magnitude = np.abs(fft_result)
        
        for start, end in flatten_ranges:
            if start >= end:
                continue

            start_idx = np.searchsorted(freqs, start, side='left')
            end_idx = np.searchsorted(freqs, end, side='right') - 1

            if start_idx != end_idx:
                interp_mag = interp1d(
                    [freqs[start_idx], freqs[end_idx]], 
                    [fft_magnitude[start_idx], fft_magnitude[end_idx]], 
                    kind='linear', fill_value="extrapolate"
                )
                
                indices = np.arange(start_idx, end_idx + 1)
                interpolated_magnitudes = interp_mag(freqs[indices])

                fft_result[indices] = interpolated_magnitudes * np.exp(1j * fft_phase[indices])

        modified_signal = np.fft.irfft(fft_result, n=len(signal))
        return modified_signal

    def process_signals_parallel(self, signals, flatten_ranges=[(59.5, 60.5), (69.5, 70.5)], max_workers=9):
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(lambda s: self.flatten_fft_peak(s, flatten_ranges), signals))
        return np.array(results)

    def find_crossings_for_peaks(self, signals, peak_ranges, frac=0.1, kernel_size=5):
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
            
            upstream_crossing = downstream_crossing = None
            
            for i in range(closest_to_mid, 0, -1):
                if (valid_magnitude_spectrum[i-1] - loess_smoothed[i-1]) * (valid_magnitude_spectrum[i] - loess_smoothed[i]) <= 0:
                    upstream_crossing = valid_freqs[i]
                    break

            for i in range(closest_to_mid, len(valid_freqs) - 1):
                if (valid_magnitude_spectrum[i] - loess_smoothed[i]) * (valid_magnitude_spectrum[i+1] - loess_smoothed[i+1]) <= 0:
                    downstream_crossing = valid_freqs[i]
                    break

            crossings.append((upstream_crossing or peak_start, downstream_crossing or peak_end))

        return crossings

    def widen_ranges(self, ranges_list, widen_by=1):
        return [(start - widen_by, end + widen_by) for start, end in ranges_list]

    def clean_and_process_ecg_leads(self, filtered_UKB, window_size=5, std_threshold=5):
        peaks_UKB, peak_freqs_UKB, peak_magnitudes_UKB, merged_regions_UKB, region_averages_UKB, harmonics_UKB, real_range_UKB = self.detect_peaks_with_sliding_window(
            np.squeeze(filtered_UKB[:, :, 0]),
            window_size=window_size,
            std_threshold=std_threshold
        )

        print(real_range_UKB)

        def process_lead(lead):
            crossings = self.find_crossings_for_peaks(np.squeeze(filtered_UKB[:, :, lead]), peak_ranges=real_range_UKB)
            crossings = self.widen_ranges(crossings)
            print(f"Lead {lead} crossings:", crossings)

            return self.process_signals_parallel(filtered_UKB[:, :, lead].astype(np.float32), flatten_ranges=crossings)

        with ThreadPoolExecutor() as executor:
            ukb_completely_clean = list(tqdm(executor.map(process_lead, range(12)), total=12))

        ukb_completely_clean = np.array(ukb_completely_clean).astype(np.float32)
        ukb_completely_clean = np.transpose(ukb_completely_clean, (1, 2, 0))
        return ukb_completely_clean