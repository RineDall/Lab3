import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcol
color_dict = mcol.TABLEAU_COLORS

def lin2db(x):
    return 10 * np.log10(x)

def db2lin(x):
    return 10 ** (x / 10)

class ADC:
    fs_step = 2.75625e3

    def __init__(s, n_bit):
        s.n_bit = n_bit

    def snr(s):
        return 6 * s.n_bit

class BSC:
    def __init__(s, error_prob):
        s.error_prob = error_prob

    def snr(s):
        if np.all(s.error_prob == 0):
            raise ValueError("Error probability must not be 0.")

        # Calculate SNR
        return 1 / (4 * s.error_prob)

def exercise_1():
    n_bits = [2, 3, 4, 6, 8, 10, 12, 14, 16]
    error_prob = np.logspace(-12, 0, 100)

    adc_snrs = [ADC(n_bit).snr() for n_bit in n_bits]
    bsc_snrs = BSC(error_prob).snr()

    plt.figure(figsize=(10, 6))
    plt.plot(n_bits, adc_snrs, label='Quantization SNR (ADC)', color='red', marker='o')
    plt.xlabel('Number of Bits (ADC)')
    plt.ylabel('SNR (dB)')
    plt.title('Quantization SNR (ADC)')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(error_prob, bsc_snrs, label='BSC SNR', color='blue')
    plt.xscale("log")
    plt.xlabel('Error Probability (BSC)')
    plt.ylabel('SNR (dB)')
    plt.title('SNR (BSC)')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    print("EXERCISE 1 LAB 3")
    exercise_1()

