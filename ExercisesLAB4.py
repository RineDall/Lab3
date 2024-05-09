import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcol
import warnings
from scipy.special import erfc

# Suppress specific warnings
warnings.filterwarnings("ignore", message="Blended transforms not yet supported.")
warnings.filterwarnings("ignore", message="divide by zero encountered in log10")

# Define color dictionary
color_dict = mcol.TABLEAU_COLORS
colors = []
color_name = []
for i, (name, color) in enumerate(color_dict.items()):
    colors.append(color)
    color_name.append(name)

def lin2db(x):
    return 10 * np.log10(x)

def db2lin(x):
    return 10 ** (x / 10)

class ADC:
    fs_step = 2.75625e3

    def __init__(self, n_bit):
        self.n__bit = np.array(n_bit, dtype=np.int64)

    @property
    def n_bit(self):
        return self.n__bit

    @n_bit.setter
    def n_bit(self, n_bit):
        self.n__bit = n_bit

    def snr(self):
        return db2lin(6 * self.n_bit)

    def sampling_freq_coefficient(self, analog_bandwidth):
        fs_step = self.fs_step
        mult = np.ceil(2 * analog_bandwidth / fs_step)
        return mult


class BSC:

    def __init__(self, error_prob):
        self.err_prob = error_prob

    @property
    def error_prob(self):
        return self.err_prob

    @error_prob.setter
    def error_probability(self, error_prob):
        self.err_prob = error_prob

    def snr(self):
        return 1 / (4 * self.error_prob)


class PCM:
    analog_bandwidth = 22e3

    def __init__(self, adc, dsi, line):  # writing digital system information as DSI
        self.adc_ = adc
        self.dsi_ = dsi
        self.line_ = line

    @property
    def adc(self):
        return self.adc_

    @adc.setter
    def adc(self, adc):
        self.adc_ = adc

    @property
    def dsi(self):
        return self.dsi_

    @dsi.setter
    def dsi(self, dsi):
        self.dsi_ = dsi

    @property
    def line(self):
        return self.line_

    @line.setter
    def line(self, line):
        self.line_ = line

    def crit_pe(self):
        m = 2 ** self.adc.n_bit
        return 1 / (4 * (m ** 2 - 1))

    def snr(self):
        return db2lin(self.line.snr_digital(self.dsi.signal_power))

    def ber_evaluation(self):
        if self.dsi.n_bit_mod == 1:
            return (1 / 2) * np.float64(erfc(np.sqrt(self.snr())))
        elif self.dsi.n_bit_mod == 2:
            return (1 / 2) * np.float64(erfc(np.sqrt(self.snr() / 2)))
        elif self.dsi.n_bit_mod == 3:
            return (2 / 3) * np.float64(erfc(np.sqrt((3 / 14) * self.snr())))
        elif self.dsi.n_bit_mod == 4:
            return (3 / 8) * np.float64(erfc(np.sqrt(self.snr() / 10)))
        else:
            # Unsupported modulation format
            print("Unsupported modulation format")
            return None


class Digital_signal_information:
    def __init__(self, signal_power, noise_power, n_bit_mod):
        self.signal_pwr = signal_power
        self.noise_pwr = noise_power
        self.n_bit_mod_ = n_bit_mod

    @property
    def signal_power(self):
        return self.signal_pwr

    @signal_power.setter
    def signal_power(self, signal_power):
        if not isinstance(signal_power, np.float64):
            raise TypeError("Signal power must be a floating point number")
        self.signal_pwr = signal_power

    @property
    def noise_power(self):
        return self.noise_pwr

    @noise_power.setter
    def noise_power(self, noise_power):
        if not isinstance(noise_power, np.float64):
            raise TypeError("Noise power must be a floating point number")
        self.noise_pwr = noise_power

    @property
    def n_bit_mod(self):
        return self.n_bit_mod_

    @n_bit_mod.setter
    def n_bit_mod(self, n_bit_mod):
        if not isinstance(n_bit_mod, int):
            raise TypeError("Number of bits must be an integer")
        self.n_bit_mod_ = n_bit_mod


class Line:
    def __init__(self, loss_coefficient, length):
        self.loss_coeff = loss_coefficient
        self.len = length

    @property
    def loss(self):
        return self.loss_coefficient * self.length / 1e3

    @property
    def length(self):
        return self.len

    @length.setter
    def length(self, length):
        if not isinstance(length, np.float64):
            raise TypeError("Length must be a floating point number")
        self.len = length

    @property
    def loss_coefficient(self):
        return self.loss_coeff

    @loss_coefficient.setter
    def loss_coefficient(self, loss_coefficient):
        if not isinstance(loss_coefficient, np.float64):
            raise TypeError("Loss coefficient must be a floating point number")
        self.loss_coeff = loss_coefficient

    def noise_generation(self, signal_power):
        return 1e-9 * signal_power * self.length

    def snr_digital(self, signal_power):
        snrd = lin2db(signal_power) - lin2db(self.noise_generation(signal_power)) - self.loss
        return snrd

def ex4_and_5_1_2():
    signal_power = 1e-3
    length = np.linspace(10, 120, 1000, dtype=np.float64)
    length = 1e3 * length
    alfa = 1
    n_bit_mod = [1, 2, 3, 4]
    n_bit_adc = 6
    snr_min = []
    l_max = []
    adc = ADC(n_bit_adc)
    line = Line(alfa, length)
    noise_power = line.noise_generation(signal_power)
    modulations = ["BPSK", "QPSK", "8QAM", "16QAM"]
    for n in range(len(n_bit_mod)):
        dsi = Digital_signal_information(signal_power, noise_power, n_bit_mod[n])
        pcm = PCM(adc, dsi, line)
        snr = lin2db(pcm.snr())
        ber = np.log10(pcm.ber_evaluation(), dtype=np.float64)
        ber_th = np.log10(pcm.crit_pe())
        index = np.where(ber < ber_th)[0][-1]
        snr_min.append(snr[index])
        l_max.append(length[index])
        plt.figure(num=1)
        plt.plot(snr, ber, color=colors[n], label=f'{modulations[n]} ({color_name[n][5:]})')
        plt.xscale("log")
        plt.axhline(y=ber_th, color="red", linestyle='--')
        plt.title("BER vs SNR")
        plt.xlabel("SNR (dB)")
        plt.ylabel("log10(BER)")
        plt.legend()
        plt.grid()

        plt.figure(num=2)
        plt.plot(length, ber, color=colors[n], label=f'{modulations[n]} ({color_name[n][5:]})')
        plt.xscale("log")
        plt.axhline(y=ber_th, color="red", linestyle='--')
        plt.title("BER vs Line Length")
        plt.xlabel("Line Length (m)")
        plt.ylabel("log10(BER)")
        plt.legend()
        plt.grid()

    plt.show()

def ex_5_3():
    signal_power = 1e-3
    length = np.linspace(10, 120, 1000, dtype=np.float64)
    length = 1e3 * length
    alfa = 1
    line = Line(alfa, length)
    noise_power = line.noise_generation(signal_power)
    n_bit_adc = [4, 6, 8, 10]
    n_bit_mod = 4
    snr_min = []
    l_max = []
    adc = ADC(n_bit_adc)
    dsi = Digital_signal_information(signal_power, noise_power, n_bit_mod)
    pcm = PCM(adc, dsi, line)
    snr = lin2db(pcm.snr())
    ber = np.log10(pcm.ber_evaluation(), dtype=np.float64)
    ber_th = np.log10(pcm.crit_pe())
    for j in range(len(n_bit_adc)):
        index = np.where(ber < ber_th[j])[0][-1]
        snr_min.append(snr[index])
        l_max.append(length[index])
        plt.figure(num=1)
        plt.plot(snr, ber, color=colors[0])
        plt.axhline(y=ber_th[j], color=colors[j + 1], linestyle='--',
                    label=f'{n_bit_adc[j]}bits ({color_name[j + 1][4:]})')
        plt.ylim([-10, 1])
        plt.xscale("log")
        plt.title("Effect of different ADC bits (BER vs SNR plot)")
        plt.xlabel("SNR (dB)")
        plt.ylabel("log10(BER)")
        plt.legend()
        plt.grid()
        plt.figure(num=2)
        plt.plot(length, ber, color=colors[0])
        plt.axhline(y=ber_th[j], color=colors[j + 1], linestyle='--',
                    label=f'{n_bit_adc[j]}bits ({color_name[j + 1][5:]})')
        plt.ylim([-10, 1])
        plt.xscale("log")
        plt.title("Effect of different ADC bits (BER vs Length plot)")
        plt.xlabel("Length (m)")
        plt.ylabel("log10(BER)")
        plt.legend()
        plt.grid()

    plt.show()

def main():
    print("SELECT:")
    print("1) BER vs SNR and Length for different modulation formats")
    print("2) Effect of ADC bits on BER vs SNR and Length")
    print("3) Exit, Do not to see anything:)")
    while True:
        choice = input("Enter your choice (1/2/3): ")
        if choice == '1':
            ex4_and_5_1_2()
        elif choice == '2':
            ex_5_3()
        elif choice == '3':
            print("Waiting for exit")
            break
        else:
            print("Invalid choice. Enter aother one :).")

if __name__ == "__main__":
    main()
