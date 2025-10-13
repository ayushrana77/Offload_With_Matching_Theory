"""
Fog Device Preferences Generator
Generates preference lists for fog devices using Research Paper Theoretical Formula:
D_i(j) = 1/(ω_j^i(ζ) + ξ_j^i + t_j,i)

Based on Formula 5 from the research paper where:
- ω_j^i(ζ): Expected waiting time for task j at fog node i
- ξ_j^i: Communication delay from task j to fog node i  
- t_j,i: Computation time for task j at fog node i
"""

import random
from typing import Dict, List


class FogPreferencesGenerator:
    def __init__(self):
        """Initialize the fog preferences generator"""
        self.fog_devices = []
        self.iot_devices = []
        self.fog_quotas = {}
    
    def set_devices(self, fog_devices: List[str], iot_devices: List[str], fog_quotas: Dict[str, int]):
        """Set the list of fog devices, IoT devices, and quotas"""
        self.fog_devices = fog_devices
        self.iot_devices = iot_devices
        self.fog_quotas = fog_quotas
    
    def generate_theoretical_server_preferences(self, servers: List[Dict], tasks: List[Dict], users: List[Dict],
                                              transmission_delays: any, server_waiting_times: Dict[str, float],
                                              server_capacities: Dict[str, int]) -> Dict[str, List[str]]:
        """
        Generate server preferences using PAPER THEORETICAL FORMULA:
        D_i(j) = 1/(ω_j^i(ζ) + ξ_j^i + t_j,i)
        
        This is Formula 5 from the research paper. Servers prefer tasks with lower overall 
        workload considering waiting time, communication delay, and computation time.
        
        Args:
            servers: List of server dictionaries with computational capabilities
            tasks: List of task dictionaries with requirements and user associations
            users: List of user dictionaries with positions
            transmission_delays: Matrix of communication delays ξ_j^i
            server_waiting_times: Current waiting times ω_j^i(ζ) for each server
            server_capacities: Capacity constraints for each server
            
        Returns:
            Dictionary mapping server_id to ranked list of task_ids (highest utility first)
        """
        from utility import PreferenceUtilities
        
        print("\n=== Generating Server Preferences (Paper Formula 5) ===")
        print("Formula: D_i(j) = 1/(omega_j^i(zeta) + xi_j^i + t_j,i)")
        print("Where:")
        print("  omega_j^i(zeta) = Expected waiting time for task j at server i")
        print("  xi_j^i = Communication delay from task j to server i")
        print("  t_j,i = Computation time for task j at server i")
        
        server_preferences = {}
        
        for server in servers:
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
                
                # Apply Paper Theoretical Formula: D_i(j) = 1/(ω_j^i(ζ) + ξ_j^i + t_j,i)
                denominator = omega_ji_zeta + xi_ji + t_ji + 1e-6  # Small epsilon to avoid division by zero
                utility_score = 1.0 / denominator
                
                task_scores[task['id']] = utility_score
                
                print(f"    {server['id']} -> {task['id']}: omega={omega_ji_zeta:.4f}, xi={xi_ji:.4f}, t={t_ji:.4f}, D={utility_score:.4f}")
            
            # Sort tasks by utility score (highest first - servers prefer better utility tasks)
            task_items = [(tid, score) for tid, score in task_scores.items()]
            sorted_tasks = PreferenceUtilities.sort_by_preference(
                task_items, lambda x: x[1], reverse=True
            )
            server_preferences[server['id']] = [task_id for task_id, _ in sorted_tasks]
            
            print(f"  {server['id']} final preferences: {' > '.join(server_preferences[server['id']])}")
        
        return server_preferences


if __name__ == "__main__":
    """Example usage of the theoretical fog preferences generator"""
    print("=== Fog Preferences Generator (Research Paper Formula) ===")
    print("This generator uses Formula 5: D_i(j) = 1/(ω_j^i(ζ) + ξ_j^i + t_j,i)")
    print("For actual usage, call generate_theoretical_server_preferences() method")
    print("with appropriate server, task, user, and system data.")