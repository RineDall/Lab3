import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcol

color_dict = mcol.TABLEAU_COLORS

class ADC:
    fs_step = 2.75625e3

    @staticmethod
    def sampling_frequency(analog_bandwidth):
        return np.ceil(analog_bandwidth / ADC.fs_step) * ADC.fs_step

    def __init__(self, n_bit):
        self.n_bit = n_bit

    def snr(self):
        return 6 * self.n_bit

class BSC:
    def __init__(self, error_probability):
        self.error_probability = error_probability

    def can_support(self, n_bit):
        return self.error_probability <= 1 / (2 ** n_bit)

class PCM:
    def __init__(self, analog_bandwidth, adc_n_bit, bsc_error_probability):
        self.analog_bandwidth = analog_bandwidth
        self.adc = ADC(adc_n_bit)
        self.bsc = BSC(bsc_error_probability)

    def effective_quantization_snr(self):
        return self.adc.snr()

# Given data
analog_bandwidth = 22e3  # 22 KHz
quantization_snr_required = 80  # dB
bsc_error_probability = 3.8e-7

print("Given Data:")
print("Analog Bandwidth:", analog_bandwidth)
print("Quantization SNR Required:", quantization_snr_required, "dB")
print("BSC Error Probability:", bsc_error_probability)

# Calculate ADC number of bits
n_bit = np.ceil(quantization_snr_required / 6).astype(int)

print("\nCalculated ADC Number of Bits:", n_bit)

# Calculate required ADC sampling frequency
adc_sampling_frequency = ADC.sampling_frequency(analog_bandwidth)

print("Required ADC Sampling Frequency:", adc_sampling_frequency, "Hz")

# Create PCM object
pcm = PCM(analog_bandwidth, n_bit, bsc_error_probability)

# Calculate effective quantization SNR
effective_quantization_snr = pcm.effective_quantization_snr()

print("Effective Quantization SNR:", effective_quantization_snr, "dB")

# Check if BSC can support transmission
can_support = pcm.bsc.can_support(n_bit)

print("Can BSC Support Transmission with the Given Number of Bits:", can_support)
