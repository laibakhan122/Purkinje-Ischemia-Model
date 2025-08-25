import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

class AdvancedPurkinjeNetwork:
    def __init__(self, n_cells=500, volume_size=100):
        self.n_cells = n_cells
        self.volume_size = volume_size
        self.cells = self._initialize_cells()
        self.adjacency_matrix = self._initialize_connectivity()

    def _initialize_cells(self):
        np.random.seed(42)
        cells = pd.DataFrame({
            'x': np.random.uniform(0, self.volume_size, self.n_cells),
            'y': np.random.uniform(0, self.volume_size, self.n_cells),
            'z': np.random.uniform(0, self.volume_size, self.n_cells),
            'type': np.random.choice(['bundle_branch', 'fascicle', 'purkinje'], self.n_cells),
            'pacemaker': np.random.choice([True, False], self.n_cells, p=[0.05, 0.95]),
            'conduction_velocity': np.random.uniform(0.3, 2.0, self.n_cells),
            'refractory_period': np.random.uniform(150, 350, self.n_cells),
            'ATP': np.random.uniform(0.7, 1.0, self.n_cells),
            'is_ischemic': False,
            'ischemia_severity': 0.0
        })
        return cells

    def _initialize_connectivity(self):
        mat = np.random.choice([0, 1], size=(self.n_cells, self.n_cells), p=[0.9, 0.1])
        np.fill_diagonal(mat, 0)
        return mat

    def apply_ischemia(self, center, radius, severity=1.0, gradient='linear'):
        coords = self.cells[['x', 'y', 'z']].values
        dist = np.linalg.norm(coords - np.array(center), axis=1)
        affected = dist <= radius

        if gradient == 'linear':
            sev = severity * (1 - dist / radius)
        elif gradient == 'exponential':
            sev = severity * np.exp(-dist / (radius / 3))
        else:
            sev = np.zeros_like(dist)

        sev[sev < 0] = 0
        self.cells.loc[affected, 'is_ischemic'] = True
        self.cells.loc[affected, 'ischemia_severity'] = sev[affected]


class ComprehensivePurkinjeSimulator:
    def __init__(self, network: AdvancedPurkinjeNetwork):
        self.network = network

    def plot_3d_network(self):
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')
        colors = np.where(self.network.cells['is_ischemic'], 'red', 'blue')
        ax.scatter(self.network.cells['x'], self.network.cells['y'], self.network.cells['z'], c=colors, s=10)
        ax.set_title("3D Purkinje Network")
        plt.show()

    def plot_conduction_velocity_histogram(self):
        plt.hist(self.network.cells['conduction_velocity'], bins=20, edgecolor='black')
        plt.title("Conduction Velocity Distribution")
        plt.xlabel("Velocity (m/s)")
        plt.ylabel("Count")
        plt.show()

    def plot_ischemia_severity(self):
        plt.scatter(self.network.cells['x'], self.network.cells['y'], c=self.network.cells['ischemia_severity'], cmap='Reds')
        plt.colorbar(label="Ischemia Severity")
        plt.title("Ischemia Severity Map (XY Projection)")
        plt.show()

    def plot_connectivity_heatmap(self):
        sns.heatmap(self.network.adjacency_matrix[:50, :50], cmap='viridis')
        plt.title("Connectivity Heatmap (First 50 Cells)")
        plt.show()


if __name__ == "__main__":
    # Example usage
    network = AdvancedPurkinjeNetwork(n_cells=500)
    network.apply_ischemia(center=(50, 50, 50), radius=30, severity=1.0, gradient='linear')

    simulator = ComprehensivePurkinjeSimulator(network)
    simulator.plot_3d_network()
    simulator.plot_conduction_velocity_histogram()
    simulator.plot_ischemia_severity()
    simulator.plot_connectivity_heatmap()
