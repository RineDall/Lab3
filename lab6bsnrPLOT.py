import numpy as np
import json
import matplotlib.pyplot as plt
import random
from scipy.special import erfcinv


class SignalInformation:
    def __init__(self, signal_power, path):
        self.signal_power = signal_power
        self.noise_power = 0.0
        self.latency = 0.0
        self.path = path

    @property
    def signal_power(self):
        return self._signal_power

    @signal_power.setter
    def signal_power(self, signal_power):
        if not isinstance(signal_power, float):
            raise TypeError("Signal power must be a floating point number")
        self._signal_power = signal_power

    @property
    def noise_power(self):
        return self._noise_power

    @noise_power.setter
    def noise_power(self, noise_power):
        if not isinstance(noise_power, float):
            raise TypeError("Noise power must be a floating point number")
        self._noise_power = noise_power

    @property
    def latency(self):
        return self._latency

    @latency.setter
    def latency(self, latency):
        if not isinstance(latency, float):
            raise TypeError("Latency must be a floating point number")
        self._latency = latency

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, path):
        if not isinstance(path, list) or not all(isinstance(node, str) for node in path):
            raise TypeError("Path must be a list of strings")
        self._path = path

    def update_signal_power(self, increment):
        self.signal_power += increment

    def update_noise_power(self, increment):
        self.noise_power += increment

    def update_latency(self, increment):
        self.latency += increment

    def update_path(self, node):
        self.path.append(node)


class Node:
    def __init__(self, node_dict):
        self.label = node_dict['label']
        self.position = tuple(map(float, node_dict.get('position', (0.0, 0.0))))
        self.connected_nodes = node_dict['connected_nodes']
        self.successive = {}
        self.transceiver = node_dict.get('transceiver', 'fixed-rate')

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, label):
        if not isinstance(label, str):
            raise TypeError("Label must be a string")
        self._label = label

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, position):
        if not (isinstance(position, tuple) and len(position) == 2 and
                all(isinstance(coord, float) for coord in position)):
            raise TypeError("Position must be a tuple of two floats")
        self._position = position

    @property
    def connected_nodes(self):
        return self._connected_nodes

    @connected_nodes.setter
    def connected_nodes(self, connected_nodes):
        if not isinstance(connected_nodes, list):
            raise TypeError("Connected nodes must be a list of strings")
        self._connected_nodes = connected_nodes

    @property
    def successive(self):
        return self._successive

    @successive.setter
    def successive(self, successive):
        if not isinstance(successive, dict):
            raise TypeError("Successive must be a dictionary")
        self._successive = successive

    def propagate(self, signal_information):
        if not signal_information.path:
            return signal_information
        next_node_label = signal_information.path.pop(0)
        if next_node_label in self.successive:
            line = self.successive[next_node_label]
            signal_information.update_path(self.label)
            return line.propagate(signal_information)
        return signal_information


class Connection:
    def __init__(self, input_node, output_node, signal_power):
        self.input = input_node
        self.output = output_node
        self.signal_power = signal_power
        self.latency = 0.0
        self.snr = 0.0
        self.bit_rate = 0.0


class Line:
    def __init__(self, label, node1, node2):
        self.label = label
        self.length = self.calculate_length(node1.position, node2.position)
        self.successive = {}
        self.state = 1  # 1 for 'free', 0 for 'occupied'

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, label):
        if not isinstance(label, str):
            raise TypeError("Label must be a string")
        self._label = label

    @property
    def length(self):
        return self._length

    @length.setter
    def length(self, length):
        if not isinstance(length, float):
            raise TypeError("Length must be a floating point number")
        self._length = length

    @property
    def successive(self):
        return self._successive

    @successive.setter
    def successive(self, successive):
        if not isinstance(successive, dict):
            raise TypeError("Successive must be a dictionary")
        self._successive = successive

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, state):
        if state not in [0, 1]:
            raise ValueError("State must be 0 (occupied) or 1 (free)")
        self._state = state

    def calculate_length(self, pos1, pos2):
        return np.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

    def latency_generation(self):
        return self.length / (2 / 3 * 3e8)

    def noise_generation(self, signal_power):
        noise_power = 1e-9 * signal_power * self.length
        min_noise_power = 1e-15  # Adjust as needed
        return max(noise_power, min_noise_power)

    def propagate(self, signal_information):
        signal_information.update_noise_power(self.noise_generation(signal_information.signal_power))
        signal_information.update_latency(self.latency_generation())
        if signal_information.path:
            next_node_label = signal_information.path.pop(0)
            if next_node_label in self.successive:
                next_node = self.successive[next_node_label]
                return next_node.propagate(signal_information)
        return signal_information


class Network:
    def __init__(self, json_file):
        self.nodes = {}
        self.lines = {}
        self.load_network(json_file)
        self.connect()

    def load_network(self, json_file):
        with open(json_file, 'r') as f:
            data = json.load(f)
        for node_label, node_info in data.items():
            self.nodes[node_label] = Node({
                'label': node_label,
                'position': node_info.get('position', [0.0, 0.0]),
                'connected_nodes': node_info['connected_nodes'],
                'transceiver': node_info.get('transceiver', 'fixed-rate')
            })
        for node_label, node in self.nodes.items():
            for connected_node_label in node.connected_nodes:
                line_label = f"{node_label}{connected_node_label}"
                if line_label not in self.lines:
                    self.lines[line_label] = Line(line_label, self.nodes[node_label], self.nodes[connected_node_label])

    def connect(self):
        for node_label, node in self.nodes.items():
            for connected_node_label in node.connected_nodes:
                line_label = f"{node_label}{connected_node_label}"
                if line_label in self.lines:
                    node.successive[connected_node_label] = self.lines[line_label]
                    self.lines[line_label].successive[connected_node_label] = self.nodes[connected_node_label]

    def find_paths(self, start_node, end_node, path=None):
        if path is None:
            path = []
        path = path + [start_node]
        if start_node == end_node:
            return [path]
        if start_node not in self.nodes:
            return []
        paths = []
        for node in self.nodes[start_node].connected_nodes:
            line_label = f"{start_node}{node}"
            line = self.lines.get(line_label)
            if line and line.state == 1:  # Check if line is free
                if node not in path:
                    new_paths = self.find_paths(node, end_node, path)
                    for new_path in new_paths:
                        paths.append(new_path)
        return paths

    def propagate(self, signal_information):
        start_node_label = signal_information.path.pop(0)
        start_node = self.nodes[start_node_label]
        return start_node.propagate(signal_information)

    def find_best_snr(self, input_node, output_node):
        paths = self.find_paths(input_node, output_node)
        best_snr = float('-inf')
        best_path = None
        for path in paths:
            path_copy = path[:]
            signal_information = SignalInformation(1e-3, path_copy)
            signal_information = self.propagate(signal_information)
            snr = 10 * np.log10(signal_information.signal_power / signal_information.noise_power)
            if snr > best_snr:
                best_snr = snr
                best_path = path
        return best_path, best_snr

    def find_best_latency(self, input_node, output_node):
        paths = self.find_paths(input_node, output_node)
        best_latency = float('inf')
        best_path = None
        for path in paths:
            latency = 0
            for i in range(len(path) - 1):
                line_label = f"{path[i]}{path[i + 1]}"
                latency += self.lines[line_label].latency_generation()
            if latency < best_latency:
                best_latency = latency
                best_path = path
        return best_path, best_latency

    def find_best_bit_rate(self, input_node, output_node):
        paths = self.find_paths(input_node, output_node)
        best_bit_rate = 0
        best_path = None
        for path in paths:
            path_copy = path[:]
            signal_information = SignalInformation(1e-3, path_copy)
            signal_information = self.propagate(signal_information)
            snr = 10 * np.log10(signal_information.signal_power / signal_information.noise_power)
            bit_rate = 2 * 32e9 * np.log2(1 + snr * 32e9 / 12.5e9)
            if bit_rate > best_bit_rate:
                best_bit_rate = bit_rate
                best_path = path
        return best_path, best_bit_rate

    def lin2db(self, x):
        return 10 * np.log10(x)

    def transceiver_strategy_plot(self):
        gsnr = np.linspace(1, 1e4, 10000)  # Generate GSNR values from 1 to 10000
        rs = 32e9  # Symbol rate (32 GHz)
        bn = 12.5e9  # Noise bandwidth (12.5 GHz)
        ber_t = 1e-3  # Target bit error rate

        # Initialize arrays to store bit rates for each strategy
        rb_fixed = np.zeros_like(gsnr)
        rb_flex = np.zeros_like(gsnr)
        rb_shannon = 2 * rs * np.log2(1 + gsnr * rs / bn)

        # Calculate bit rate for fixed-rate strategy
        for idx in range(len(gsnr)):
            if gsnr[idx] >= 2 * (erfcinv(2 * ber_t)) ** 2 * (rs / bn):
                rb_fixed[idx] = 100e9
            else:
                rb_fixed[idx] = 0

        # Calculate bit rate for flex-rate strategy
        for idx in range(len(gsnr)):
            if gsnr[idx] < 2 * (erfcinv(2 * ber_t)) ** 2 * (rs / bn):
                rb_flex[idx] = 0
            elif gsnr[idx] < (14 / 3) * (erfcinv(3 / 2 * ber_t)) ** 2 * (rs / bn):
                rb_flex[idx] = 100e9
            elif gsnr[idx] < 10 * (erfcinv(3 / 2 * ber_t)) ** 2 * (rs / bn):
                rb_flex[idx] = 200e9
            else:
                rb_flex[idx] = 400e9

        # Plotting
        plt.figure(figsize=(8, 6))
        plt.semilogy(self.lin2db(gsnr), rb_fixed / 1e9, label='Fixed-Rate', color='blue')
        plt.semilogy(self.lin2db(gsnr), rb_flex / 1e9, label='Flex-Rate', color='red')
        plt.semilogy(self.lin2db(gsnr), rb_shannon / 1e9, label='Shannon', color='green')
        plt.legend(loc='best')
        plt.xlabel('GSNR (dB)')
        plt.ylabel('Bit-Rate (Gbps)')
        plt.title('Bit Rate vs. GSNR')
        plt.grid(True)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":

    network = Network('nodes (1).json')

    # Example of using find_best_bit_rate method
    input_node = 'A'
    output_node = 'F'
    best_path, best_bit_rate = network.find_best_bit_rate(input_node, output_node)
    print(f"Best path for maximum bit rate from {input_node} to {output_node}: {best_path}")
    print(f"Maximum bit rate: {best_bit_rate / 1e9} Gbps")

    # Plot the bit rate vs. GSNR curve
    network.transceiver_strategy_plot()
