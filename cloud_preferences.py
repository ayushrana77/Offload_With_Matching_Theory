"""
Cloud Device Preferences Generator
Generates preference lists for cloud devices using Research Paper Theoretical Formula:
D_i(j) = 1/(ω_j^i(ζ) + ξ_j^i + t_j,i + c_i)

Based on Formula 5 from the research paper with cloud-specific modifications where:
- ω_j^i(ζ): Expected waiting time for task j at cloud node i
- ξ_j^i: Communication delay from task j to cloud node i
- t_j,i: Computation time for task j at cloud node i
- c_i: Cloud cost factor (higher for expensive cloud resources)
"""

import random
from typing import Dict, List


class CloudPreferencesGenerator:
    def __init__(self):
        """Initialize the cloud preferences generator"""
        self.cloud_devices = []
        self.iot_devices = []
        self.cloud_quotas = {}

    def set_devices(self, cloud_devices: List[str], iot_devices: List[str], cloud_quotas: Dict[str, int]):
        """Set the list of cloud devices, IoT devices, and quotas"""
        self.cloud_devices = cloud_devices
        self.iot_devices = iot_devices
        self.cloud_quotas = cloud_quotas

    def generate_theoretical_cloud_preferences(self, servers: List[Dict], tasks: List[Dict], users: List[Dict],
                                             transmission_delays: any, server_waiting_times: Dict[str, float],
                                             server_capacities: Dict[str, int]) -> Dict[str, List[str]]:
        """
        Generate cloud server preferences using PAPER THEORETICAL FORMULA with cloud modifications:
        D_i(j) = 1/(ω_j^i(ζ) + ξ_j^i + t_j,i + c_i)

        This is Formula 5 with cloud cost factor. Cloud servers prioritize computationally intensive
        tasks that can benefit from unlimited cloud resources, while accounting for higher costs.

        Args:
            servers: List of cloud server dictionaries with computational capabilities
            tasks: List of task dictionaries with requirements and user associations
            users: List of user dictionaries with positions
            transmission_delays: Matrix of communication delays ξ_j^i
            server_waiting_times: Current waiting times ω_j^i(ζ) for each server
            server_capacities: Capacity constraints for each server

        Returns:
            Dictionary mapping server_id to ranked list of task_ids (highest utility first)
        """
        from utility import PreferenceUtilities

        print("\n=== Generating Cloud Server Preferences (Paper Formula 5 + Cost Factor) ===")
        print("Formula: D_i(j) = 1/(omega_j^i(zeta) + xi_j^i + t_j,i + c_i)")
        print("Where:")
        print("  omega_j^i(zeta) = Expected waiting time for task j at cloud server i")
        print("  xi_j^i = Communication delay from task j to cloud server i")
        print("  t_j,i = Computation time for task j at cloud server i")
        print("  c_i = Cloud cost factor (higher for expensive cloud resources)")

        server_preferences = {}

        for server in servers:
            # Only process cloud servers (level 3)
            if server.get('level') != 3:
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

                # Formula Component 4: c_i (Cloud cost factor)
                # Cloud servers have higher operational costs but better computational capabilities
                # Cost factor considers processing cost and energy efficiency
                processing_cost = server.get('processing_cost', 0.05)  # Default cloud cost
                energy_efficiency = server.get('energy_efficiency', 0.9)

                # Higher computational requirements get cost bonus in cloud (economies of scale)
                comp_requirement_factor = min(1.0, task['computation_requirement'] / 1e9)  # Normalize by 1 GFLOP
                cost_bonus = comp_requirement_factor * 0.1  # Up to 0.1 bonus for compute-intensive tasks

                c_i = processing_cost / energy_efficiency - cost_bonus

                # Apply Paper Theoretical Formula with cloud cost: D_i(j) = 1/(ω_j^i(ζ) + ξ_j^i + t_j,i + c_i)
                denominator = omega_ji_zeta + xi_ji + t_ji + c_i + 1e-6  # Small epsilon to avoid division by zero
                utility_score = 1.0 / denominator

                task_scores[task['id']] = utility_score

                print(f"    {server['id']} -> {task['id']}: omega={omega_ji_zeta:.4f}, xi={xi_ji:.4f}, t={t_ji:.4f}, c={c_i:.4f}, D={utility_score:.4f}")

            # Sort tasks by utility score (highest first - servers prefer better utility tasks)
            task_items = [(tid, score) for tid, score in task_scores.items()]
            sorted_tasks = PreferenceUtilities.sort_by_preference(
                task_items, lambda x: x[1], reverse=True
            )
            server_preferences[server['id']] = [task_id for task_id, _ in sorted_tasks]

            print(f"  {server['id']} final preferences: {' > '.join(server_preferences[server['id']])}")

        return server_preferences


if __name__ == "__main__":
    """Example usage of the theoretical cloud preferences generator"""
    print("=== Cloud Preferences Generator (Research Paper Formula + Cost) ===")
    print("This generator uses Formula 5: D_i(j) = 1/(ω_j^i(ζ) + ξ_j^i + t_j,i + c_i)")
    print("With cloud cost factor for resource-efficient computing")
    print("For actual usage, call generate_theoretical_cloud_preferences() method")
    print("with appropriate server, task, user, and system data.")