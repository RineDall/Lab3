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

class PCM:
    def __init__(s, analog_bandwidth, adc_n_bit, bsc_error_prob):
        s.analog_bandwidth = analog_bandwidth
        s.adc = ADC(adc_n_bit)
        s.bsc = BSC(bsc_error_prob)

    def snr(self):
        snr_adc = self.adc.snr()
        snr_bsc = self.bsc.snr()
        snr_total = 1 / (1 / snr_adc + 1 / snr_bsc)
        return snr_total

    #another comment
    def critical_pe(s):
        snr_adc = s.adc.snr()
        snr_bsc = s.bsc.snr()
        pe_critical = 1 / (1 / snr_adc + 1 / snr_bsc)
        return pe_critical[0]

def exercise_2():
    n_bits = [2, 4, 8, 16]
    error_prob = np.logspace(-12, 0, 100)

    plt.figure(figsize=(10, 6))

    for n_bit in n_bits:
        pcm = PCM(analog_bandwidth=1e6, adc_n_bit=n_bit, bsc_error_prob=error_prob)
        snr_total = pcm.snr()

        plt.plot(error_prob, snr_total, label=f'{n_bit} bits', marker='o')

        critical_pe = pcm.critical_pe()
        plt.axvline(x=critical_pe, color='gray', linestyle='--')
        plt.axhline(y=snr_total[0], color='black', linestyle='--')

    plt.xscale("log")
    plt.xlabel('Error Probability (BSC)')
    plt.ylabel('SNR (dB)')
    plt.title('Overall SNR vs Error Probability')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    print("EXERCISE 2 LAB 3")
    exercise_2()

