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
        self.snr = 0.0  # Initialize SNR attribute

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

    @property
    def snr(self):
        return self._snr

    @snr.setter
    def snr(self, snr):
        if not isinstance(snr, float):
            raise TypeError("SNR must be a floating point number")
        self._snr = snr

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
        self.transceiver = node_dict.get('transceiver', 'fixed-rate')  # Default to 'fixed-rate' if not provided

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

    @property
    def transceiver(self):
        return self._transceiver

    @transceiver.setter
    def transceiver(self, transceiver):
        if transceiver not in ['fixed-rate', 'flex-rate', 'shannon']:
            raise ValueError("Transceiver type must be 'fixed-rate', 'flex-rate', or 'shannon'")
        self._transceiver = transceiver


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
                next_node_or_line = self.successive[next_node_label]

                # Ensure next_node_or_line is an instance of Line
                if isinstance(next_node_or_line, Line):
                    return next_node_or_line.propagate(signal_information)
                elif isinstance(next_node_or_line, Node):
                    # Handle propagation to Node if necessary
                    # For example, you might want to update latency or noise here
                    return signal_information

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

        if signal_information.path:
            next_node_label = signal_information.path[0]
            line_label = f"{start_node_label}{next_node_label}"
            if line_label in self.lines:
                line = self.lines[line_label]
                return line.propagate(signal_information)

        return signal_information

    def find_best_snr(self, input_node, output_node):
        paths = self.find_paths(input_node, output_node)
        best_snr = float('-inf')
        best_path = None
        for path in paths:
            path_copy = path[:]
            signal_information = SignalInformation(1e-3, path_copy)
            signal_information = self.propagate(signal_information)
            snr_linear = 10 ** (signal_information.snr / 10)  # Convert SNR from dB to linear
            if snr_linear > best_snr:
                best_snr = snr_linear
                best_path = path
        return best_path, best_snr

    def propagate(self, signal_information):
        start_node_label = signal_information.path.pop(0)
        start_node = self.nodes[start_node_label]

        if signal_information.path:
            next_node_label = signal_information.path[0]
            line_label = f"{start_node_label}{next_node_label}"
            if line_label in self.lines:
                line = self.lines[line_label]
                signal_information = line.propagate(signal_information)

                # Calculate SNR and update signal_information
                signal_power = signal_information.signal_power
                noise_power = signal_information.noise_power
                snr_dB = 10 * np.log10(signal_power / noise_power)
                signal_information.snr = snr_dB

        return signal_information

    def calculate_bit_rate(self, path, connection):
        line_label = f"{path[0]}{path[1]}"
        line = self.lines[line_label]
        GSNR_dB = connection.snr - 1.6  # Assuming 1.6 dB penalty
        GSNR_linear = 10 ** (GSNR_dB / 10)
        N = 12.5e9  # Noise spectral density (W/Hz)
        Rb_fixed_rate = 100e9  # Fixed bit rate (bps)
        Rb_flex_rate = 200e9  # Flexible bit rate (bps)
        Rb_shannon = N * line.length * np.log2(1 + GSNR_linear)  # Shannon capacity (bps)

        if self.nodes[path[0]].transceiver == 'fixed-rate':
            bit_rate = Rb_fixed_rate
        elif self.nodes[path[0]].transceiver == 'flex-rate':
            bit_rate = Rb_flex_rate
        elif self.nodes[path[0]].transceiver == 'shannon':
            bit_rate = Rb_shannon
        else:
            bit_rate = 0.0

        return bit_rate

    def stream(self, connections, method='snr'):
        total_capacity = 0.0
        accepted_bit_rates = []

        for connection in connections:
            input_node = connection.input
            output_node = connection.output

            if method == 'snr':
                best_path, best_metric = self.find_best_snr(input_node, output_node)
            else:
                best_path, best_metric = self.find_best_latency(input_node, output_node)

            if best_path:
                connection.latency = self.calculate_latency(best_path)
                connection.snr = 10 * np.log10(best_metric)  # Store SNR in dB for easier interpretation
                strategy = self.nodes[best_path[0]].transceiver  # Transceiver strategy of the first node in path
                bit_rate = self.calculate_bit_rate(best_path, connection)
                connection.bit_rate = bit_rate
                total_capacity += bit_rate
                if bit_rate > 0:
                    accepted_bit_rates.append(bit_rate)

                # Print statements for debugging
                print(f"Connection from {input_node} to {output_node}:")
                print(f"   Path: {best_path}")
                print(f"   Latency: {connection.latency}")
                print(f"   SNR: {connection.snr} dB")
                print(f"   Bit Rate: {bit_rate / 1e9} Gbps")

        # Calculate overall average bit rate
        if accepted_bit_rates:
            overall_average_bit_rate = sum(accepted_bit_rates) / len(accepted_bit_rates)
        else:
            overall_average_bit_rate = 0.0

        print(f"\nOverall Average Bit Rate: {overall_average_bit_rate / 1e9} Gbps")
        print(f"Total Network Capacity: {total_capacity / 1e9} Gbps")

        return accepted_bit_rates

    def calculate_latency(self, path):
        latency = 0.0
        for i in range(len(path) - 1):
            line_label = f"{path[i]}{path[i + 1]}"
            line = self.lines[line_label]
            latency += line.latency_generation()
        return latency

    def plot_bitrate_histogram(self, bit_rates, file_name):
        bit_rates = [bit_rate for bit_rate in bit_rates if bit_rate is not None]
        plt.hist(bit_rates, bins=len(bit_rates), range=(min(bit_rates), max(bit_rates)), edgecolor='black')
        plt.xlabel('Bit Rate (Gbps)')
        plt.ylabel('Occurrences')
        plt.title('Bit Rate Distribution')
        plt.grid(True)
        plt.savefig(file_name, dpi=600)
        plt.show()


def main():
    # List of JSON files with network descriptions in this case is only one
    json_files = ['299877.json']
    transceiver_strategies = ['fixed-rate', 'flex-rate', 'shannon']

    for json_file in json_files:
        print(f"Evaluating network described in '{json_file}'...")

        # Initialize the network
        network = Network(json_file)

        # Collect accepted bit rates for each transceiver strategy
        accepted_bit_rates = {}
        total_capacity = {}

        for strategy in transceiver_strategies:
            # Generate 100 random connections
            connections = []
            random.seed(42)
            for _ in range(100):
                input_node = random.choice(list(network.nodes.keys()))
                output_node = random.choice(list(network.nodes.keys()))
                while input_node == output_node:
                    output_node = random.choice(list(network.nodes.keys()))
                connections.append(Connection(input_node, output_node, 1e-3))  # Initial signal power of 1 mW

            # Stream connections based on SNR and collect accepted bit rates
            accepted_bit_rates[strategy] = network.stream(connections, method='snr')

            # Calculate total capacity allocated in the network
            total_capacity[strategy] = sum(accepted_bit_rates[strategy])

        # Print results for the current network
        print(f"\nResults for network '{json_file}':")
        for strategy in transceiver_strategies:
            print(f"\nTransceiver Strategy: {strategy}")
            if accepted_bit_rates[strategy]:
                overall_average_bit_rate = sum(accepted_bit_rates[strategy]) / len(accepted_bit_rates[strategy])
                print(f"Overall Average Bit Rate: {overall_average_bit_rate / 1e9} Gbps")
                print(f"Total Network Capacity: {total_capacity[strategy] / 1e9} Gbps")
            else:
                print("No connections accepted.")

        # Plot histogram of accepted bit rates
        for strategy in transceiver_strategies:
            file_name = f'{json_file}_{strategy}_bitrate_distribution.png'
            network.plot_bitrate_histogram(accepted_bit_rates[strategy], file_name)


if __name__ == "__main__":
    main()
