"""
IoT Device Preferences Generator
Generates preference lists for IoT devices using Research Paper Theoretical Formula:
O_j(i) = 1/(ω_j^i(ζ) + ξ_j^i)

Based on Formula 4 from the research paper where:
- ω_j^i(ζ): Expected waiting time for task j at fog node i
- ξ_j^i: Communication delay from task j to fog node i
"""

import random
import numpy as np
from typing import Dict, List


class IoTPreferencesGenerator:
    def __init__(self):
        """Initialize the IoT preferences generator"""
        self.iot_devices = []
        self.fog_devices = []
    
    def set_devices(self, iot_devices: List[str], fog_devices: List[str]):
        """Set the list of IoT devices and fog devices"""
        self.iot_devices = iot_devices
        self.fog_devices = fog_devices
    
    def generate_theoretical_task_preferences(self, users: List[Dict], servers: List[Dict], 
                                            transmission_delays: any, server_waiting_times: Dict[str, float],
                                            server_capacities: Dict[str, int], tasks: List[Dict]) -> Dict[str, List[str]]:
        """
        Generate TASK preferences (replacing User preferences).
        Each TASK decides its own preferred server based on its specific requirements.
        
        Formula: O_j(i) = 1/(ω_j^i(ζ) + ξ_j^i) * PowerBonus
        
        Power Bonus is DYNAMIC:
        - Heavy Tasks (> 4000 M-cycles): Strong bonus for high-CPU servers (Cloud preference)
        - Light Tasks (< 4000 M-cycles): Zero/Low bonus (Edge preference via latency)
        
        Args:
            users: List of user dictionaries (for position lookup)
            servers: List of server dictionaries
            transmission_delays: Matrix of delays
            server_waiting_times: Current server waiting times
            server_capacities: Server capacities
            tasks: List of task dictionaries
            
        Returns:
            Dictionary mapping task_id to ranked list of server_ids
        """
        from .algorithm_utilities import AlgorithmUtilities
        
        print("\n=== Generating Task-Specific Preferences (Dynamic Best-Fit) ===")
        print("Strategy: Heavy Tasks -> Cloud, Light Tasks -> Edge")
        
        task_preferences = {}
        
        # Helper map for user index lookup
        user_id_to_index = {u['id']: i for i, u in enumerate(users)}
        
        for task in tasks:
            user_id = task['user_id']
            user_idx = user_id_to_index.get(user_id)
            if user_idx is None:
                continue
                
            server_scores = {}
            
            # Determine if task is "Heavy" (needs Cloud) or "Light" (needs Edge)
            # Threshold: 4000 M-cycles (midpoint of 1000-7000 range)
            is_heavy = task['computation_requirement'] > 4000e6
            
            # Dynamic Power Factor alpha
            # Heavy: 0.6 (Strong pull to Cloud)
            # Light: 0.0 (No pull -> rely on latency -> Edge)
            alpha = 0.6 if is_heavy else 0.0
            
            for j, server in enumerate(servers):
                # 1. Communication Delay
                xi_ji = transmission_delays[user_idx][j]
                
                # 2. Waiting Time
                if server['id'] in server_waiting_times:
                    omega_ji_zeta = server_waiting_times[server['id']]
                else:
                    # Initial estimate
                    current_load = 0
                    server_capacity = server_capacities.get(server['id'], 1)
                    load_factor = current_load / server_capacity
                    comp_time = task['computation_requirement'] / server['computational_capability']
                    omega_ji_zeta = load_factor + comp_time
                
                # Base Score: 1 / Time
                denominator = omega_ji_zeta + xi_ji + 1e-6
                base_score = 1.0 / denominator
                
                # 3. Dynamic Power Bonus
                # High CPU servers get a boost ONLY for heavy tasks
                cpu_ghz = server['computational_capability'] / 1e9
                power_bonus = 1.0 + (alpha * np.log10(max(1.0, cpu_ghz)))
                
                utility_score = base_score * power_bonus
                server_scores[server['id']] = utility_score
            
            # Sort servers for this task
            server_items = [(sid, score) for sid, score in server_scores.items()]
            sorted_servers = AlgorithmUtilities.sort_by_preference(
                server_items, lambda x: x[1], reverse=True
            )
            task_preferences[task['id']] = [sid for sid, _ in sorted_servers]
        
        return task_preferences

if __name__ == "__main__":
    """Example usage of the theoretical IoT preferences generator"""
    print("=== IoT Preferences Generator (Research Paper Formula) ===")
    print("This generator uses Formula 4: O_j(i) = 1/(ω_j^i(ζ) + ξ_j^i)")
    print("For actual usage, call generate_theoretical_user_preferences() method")
    print("with appropriate user, server, and system data.")