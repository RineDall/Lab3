import numpy as np

class ADC:
    fs_step = 2.75625e3

    @staticmethod
    def sampling_freq(analog_band):
        return np.ceil(analog_band / ADC.fs_step) * ADC.fs_step

    def __init__(s, n_bits):
        s.n_bit = n_bits

    def snr(s):
        return 6 * s.n_bit

class BSC:
    def __init__(s, error_prob):
        s.error_prob = error_prob

    def can_transmit(s, n_bit):
        pe_tresh = (1 / (4 * (2 ** (2 * n_bit) - 1)))
        print("Pe threshold", pe_tresh)
        return s.error_prob <= pe_tresh

class PCM:
    def __init__(s, analog_band, adc_n_bit, bsc_error_prob):
        s.analog_band = analog_band
        s.adc = ADC(adc_n_bit)
        s.bsc = BSC(bsc_error_prob)

    def effective_quant_snr(s):
        return s.adc.snr()

# Given data
analog_band = 22e3  # 22 KHz
quant_snr_required = 80  # dB
bsc_error_prob = 3.8e-7

print("Given Data:")
print("Analog Bandwidth:", analog_band)
print("Quantization SNR Required:", quant_snr_required, "dB")
print("BSC Error Probability:", bsc_error_prob)

# ADC number of bits
n_bit = np.ceil(quant_snr_required / 6).astype(int)

print("\n ADC Number of Bits:", n_bit)

# Calculating sampling freq ADC
adc_sampling_freq = ADC.sampling_freq(analog_band)

print("Required Sampling Frequency ADC:", adc_sampling_freq, "Hz")

# Creating PCM obj
pcm = PCM(analog_band, n_bit, bsc_error_prob)

# Calculating effective quantization SNR
effective_quant_snr = pcm.effective_quant_snr()

print("SNR Effective Quantization:", effective_quant_snr, "dB")

# Check if BSC can succesfully transmit
can_support = pcm.bsc.can_transmit(n_bit)

print("Can BSC with the given nr of bits transmit:", can_support)
