import numpy as np
import json
import matplotlib.pyplot as plt
import pandas as pd
import os


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
        if not isinstance(path, list):
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
        self.position = tuple(map(float, node_dict['position']))
        self.connected_nodes = node_dict['connected_nodes']
        self.successive = {}

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



class Line:
    def __init__(self, label, node1, node2):
        self.label = label
        self.length = self.calculate_length(node1.position, node2.position)
        self.successive = {}

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

    def calculate_length(self, pos1, pos2):
        return np.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

    def latency_generation(self):
        return self.length / (2 / 3 * 3e8)

    def noise_generation(self, signal_power):
        # Calculating noise power
        noise_power = 1e-9 * signal_power * self.length
        # Ensuring minimum noise power
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
                'position': node_info['position'],
                'connected_nodes': node_info['connected_nodes']
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
            if node not in path:
                new_paths = self.find_paths(node, end_node, path.copy())  # Use copy of path
                for new_path in new_paths:
                    paths.append(new_path)
        return paths

    def propagate(self, signal_information):
        start_node_label = signal_information.path.pop(0)
        if start_node_label in self.nodes:
            start_node = self.nodes[start_node_label]
            return start_node.propagate(signal_information)
        return signal_information

    def draw(self):
        plt.figure()
        for line in self.lines.values():
            node_labels = line.label
            node1_label = node_labels[0]
            node2_label = node_labels[1]
            node1_pos = self.nodes[node1_label].position
            node2_pos = self.nodes[node2_label].position
            plt.plot([node1_pos[0], node2_pos[0]], [node1_pos[1], node2_pos[1]], 'k-', lw=2)
        for node in self.nodes.values():
            plt.scatter(node.position[0], node.position[1], s=100, label=node.label)
        plt.legend()
        plt.xlabel('Position X')
        plt.ylabel('Position Y')
        plt.title('Network Topology')
        plt.show()

    def create_dataframe(self):
        data = {'Path': [], 'Latency': [], 'Noise Power': [], 'SNR': []}
        for start_node_label, start_node in self.nodes.items():
            for end_node_label, end_node in self.nodes.items():
                if start_node_label != end_node_label:
                    paths = self.find_paths(start_node_label, end_node_label)
                    for path in paths:
                        signal_info = SignalInformation(1e-3, path.copy())
                        propagated_signal = self.propagate(signal_info)
                        latency = propagated_signal.latency
                        noise_power = propagated_signal.noise_power
                        if noise_power == 0:
                            snr = float('inf')  # Handle division by zero
                        else:
                            snr = 10 * np.log10(propagated_signal.signal_power / noise_power)
                        data['Path'].append(' -> '.join(path))
                        data['Latency'].append(latency)
                        data['Noise Power'].append(noise_power)
                        data['SNR'].append(snr)
        df = pd.DataFrame(data)

        return df


if __name__ == "__main__":

    json_file = r'C:\Users\user\Downloads\nodes (1).json'
    network = Network(json_file)
    network.draw()
    df = network.create_dataframe()
    print(df)

