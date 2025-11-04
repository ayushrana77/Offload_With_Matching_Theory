"""
Edge Fog Device Preferences Generator
Generates preference lists for edge fog devices using Research Paper Theoretical Formula:
D_i(j) = 1/(ω_j^i(ζ) + ξ_j^i + t_j,i + λ_i)

Based on Formula 5 from the research paper with edge-specific modifications where:
- ω_j^i(ζ): Expected waiting time for task j at edge fog node i
- ξ_j^i: Communication delay from task j to edge fog node i
- t_j,i: Computation time for task j at edge fog node i
- λ_i: Edge proximity bonus (lower for closer edge nodes)
"""

import random
from typing import Dict, List


class EdgePreferencesGenerator:
    def __init__(self):
        """Initialize the edge preferences generator"""
        self.edge_devices = []
        self.iot_devices = []
        self.edge_quotas = {}

    def set_devices(self, edge_devices: List[str], iot_devices: List[str], edge_quotas: Dict[str, int]):
        """Set the list of edge devices, IoT devices, and quotas"""
        self.edge_devices = edge_devices
        self.iot_devices = iot_devices
        self.edge_quotas = edge_quotas

    def generate_theoretical_edge_preferences(self, servers: List[Dict], tasks: List[Dict], users: List[Dict],
                                            transmission_delays: any, server_waiting_times: Dict[str, float],
                                            server_capacities: Dict[str, int]) -> Dict[str, List[str]]:
        """
        Generate edge server preferences using PAPER THEORETICAL FORMULA with edge modifications:
        D_i(j) = 1/(ω_j^i(ζ) + ξ_j^i + t_j,i + λ_i)

        This is Formula 5 with edge proximity bonus. Edge servers prioritize tasks from nearby
        IoT devices to minimize latency and maximize edge computing benefits.

        Args:
            servers: List of edge server dictionaries with computational capabilities
            tasks: List of task dictionaries with requirements and user associations
            users: List of user dictionaries with positions
            transmission_delays: Matrix of communication delays ξ_j^i
            server_waiting_times: Current waiting times ω_j^i(ζ) for each server
            server_capacities: Capacity constraints for each server

        Returns:
            Dictionary mapping server_id to ranked list of task_ids (highest utility first)
        """
        from .algorithm_utilities import AlgorithmUtilities

        print("\n=== Generating Edge Server Preferences (Paper Formula 5 + Edge Proximity) ===")
        print("Formula: D_i(j) = 1/(omega_j^i(zeta) + xi_j^i + t_j,i + lambda_i)")
        print("Where:")
        print("  omega_j^i(zeta) = Expected waiting time for task j at edge server i")
        print("  xi_j^i = Communication delay from task j to edge server i")
        print("  t_j,i = Computation time for task j at edge server i")
        print("  lambda_i = Edge proximity bonus (lower for closer edge nodes)")

        server_preferences = {}

        for server in servers:
            # Only process edge servers (level 1)
            if server.get('level') != 1:
                continue

            task_scores = {}

            for task in tasks:
                # Find the user associated with this task
                user_index = next(i for i, u in enumerate(users) if u['id'] == task['user_id'])
                server_index = next(i for i, s in enumerate(servers) if s['id'] == server['id'])

                # Formula Component 1: ξ_j^i (Communication delay from task j's user to server i)
                xi_ji = transmission_delays[user_index][server_index]

                # Formula Component 2: ω_j^i(ζ) (Expected waiting time for task j at server i)
                if server['id'] in server_waiting_times:
                    # Use dynamically updated waiting time from algorithm (pseudocode line 17)
                    omega_ji_zeta = server_waiting_times[server['id']]
                else:
                    # Calculate initial waiting time estimate based on current load
                    current_server_load = len([t for t in tasks if t.get('assigned_server') == server['id']])
                    server_capacity = server_capacities.get(server['id'], 1)
                    load_factor = current_server_load / server_capacity
                    omega_ji_zeta = load_factor * task['deadline']  # Scale by task deadline

                # Formula Component 3: t_j,i (Computation time for task j at server i)
                t_ji = task['computation_requirement'] / server['computational_capability']

                # Formula Component 4: λ_i (Edge proximity bonus)
                # Edge servers get bonus for being close to IoT devices (reduces effective delay)
                user_pos = users[user_index]['position']
                server_pos = server['position']
                distance = ((user_pos[0] - server_pos[0])**2 + (user_pos[1] - server_pos[1])**2)**0.5

                # Proximity bonus: lower distance = lower lambda (better score)
                # Normalize by coverage radius and add small constant
                coverage_radius = server.get('coverage_radius', 50.0)
                lambda_i = max(0, (distance / coverage_radius) - 0.5) * 0.1  # 0 to 0.1 bonus range

                # Apply Paper Theoretical Formula with edge proximity: D_i(j) = 1/(ω_j^i(ζ) + ξ_j^i + t_j,i + λ_i)
                denominator = omega_ji_zeta + xi_ji + t_ji + lambda_i + 1e-6  # Small epsilon to avoid division by zero
                utility_score = 1.0 / denominator

                task_scores[task['id']] = utility_score

                print(f"    {server['id']} -> {task['id']}: omega={omega_ji_zeta:.4f}, xi={xi_ji:.4f}, t={t_ji:.4f}, lambda={lambda_i:.4f}, D={utility_score:.4f}")

            # Sort tasks by utility score (highest first - servers prefer better utility tasks)
            task_items = [(tid, score) for tid, score in task_scores.items()]
            sorted_tasks = AlgorithmUtilities.sort_by_preference(
                task_items, lambda x: x[1], reverse=True
            )
            server_preferences[server['id']] = [task_id for task_id, _ in sorted_tasks]

            print(f"  {server['id']} final preferences: {' > '.join(server_preferences[server['id']])}")

        return server_preferences


if __name__ == "__main__":
    """Example usage of the theoretical edge preferences generator"""
    print("=== Edge Preferences Generator (Research Paper Formula + Proximity) ===")
    print("This generator uses Formula 5: D_i(j) = 1/(ω_j^i(ζ) + ξ_j^i + t_j,i + λ_i)")
    print("With edge proximity bonus for low-latency computing")
    print("For actual usage, call generate_theoretical_edge_preferences() method")
    print("with appropriate server, task, user, and system data.")