"""
IoT Device Preferences Generator
Generates preference lists for IoT devices using Research Paper Theoretical Formula:
O_j(i) = 1/(ω_j^i(ζ) + ξ_j^i)

Based on Formula 4 from the research paper where:
- ω_j^i(ζ): Expected waiting time for task j at fog node i
- ξ_j^i: Communication delay from task j to fog node i
"""

import random
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
    
    def generate_theoretical_user_preferences(self, users: List[Dict], servers: List[Dict], 
                                            transmission_delays: any, server_waiting_times: Dict[str, float],
                                            server_capacities: Dict[str, int], tasks: List[Dict]) -> Dict[str, List[str]]:
        """
        Generate user preferences using PAPER THEORETICAL FORMULA:
        O_j(i) = 1/(ω_j^i(ζ) + ξ_j^i)
        
        This is Formula 4 from the research paper. Users prefer servers with lower 
        waiting time and communication delay.
        
        Args:
            users: List of user dictionaries with positions and characteristics
            servers: List of server dictionaries with capabilities and positions
            transmission_delays: Matrix of communication delays ξ_j^i
            server_waiting_times: Current waiting times ω_j^i(ζ) for each server
            server_capacities: Capacity constraints for each server
            tasks: List of task dictionaries for computation requirements
            
        Returns:
            Dictionary mapping user_id to ranked list of server_ids (highest utility first)
        """
        from utility import PreferenceUtilities
        
        print("\n=== Generating User Preferences (Paper Formula 4) ===")
        print("Formula: O_j(i) = 1/(omega_j^i(zeta) + xi_j^i)")
        print("Where:")
        print("  omega_j^i(zeta) = Expected waiting time for task j at server i")
        print("  xi_j^i = Communication delay from task j to server i")
        
        user_preferences = {}
        
        for i, user in enumerate(users):
            server_scores = {}
            
            for j, server in enumerate(servers):
                # Formula Component 1: ξ_j^i (Communication delay from user j to server i)
                xi_ji = transmission_delays[i][j]
                
                # Formula Component 2: ω_j^i(ζ) (Expected waiting time for task j at server i)
                if server['id'] in server_waiting_times:
                    # Use dynamically updated waiting time from algorithm (pseudocode line 17)
                    omega_ji_zeta = server_waiting_times[server['id']]
                else:
                    # Calculate initial waiting time estimate based on current load and task requirements
                    current_load = len([t for t in tasks if t['user_id'] == user['id']])
                    server_capacity = server_capacities.get(server['id'], 1)
                    load_factor = current_load / server_capacity
                    
                    # Add computation time component based on user's task requirements
                    user_tasks = [t for t in tasks if t['user_id'] == user['id']]
                    if user_tasks:
                        avg_comp_requirement = sum(t['computation_requirement'] for t in user_tasks) / len(user_tasks)
                        comp_time = avg_comp_requirement / server['computational_capability']
                    else:
                        comp_time = 1.0  # Default computation time
                    
                    omega_ji_zeta = load_factor + comp_time
                
                # Apply Paper Theoretical Formula: O_j(i) = 1/(ω_j^i(ζ) + ξ_j^i)
                denominator = omega_ji_zeta + xi_ji + 1e-6  # Small epsilon to avoid division by zero
                utility_score = 1.0 / denominator
                
                server_scores[server['id']] = utility_score
                
                print(f"    {user['id']} -> {server['id']}: omega={omega_ji_zeta:.4f}, xi={xi_ji:.4f}, O={utility_score:.4f}")
            
            # Sort servers by utility score (highest first - users prefer better utility servers)
            server_items = [(sid, score) for sid, score in server_scores.items()]
            sorted_servers = PreferenceUtilities.sort_by_preference(
                server_items, lambda x: x[1], reverse=True
            )
            user_preferences[user['id']] = [server_id for server_id, _ in sorted_servers]
            
            print(f"  {user['id']} final preferences: {' > '.join(user_preferences[user['id']])}")
        
        return user_preferences

if __name__ == "__main__":
    """Example usage of the theoretical IoT preferences generator"""
    print("=== IoT Preferences Generator (Research Paper Formula) ===")
    print("This generator uses Formula 4: O_j(i) = 1/(ω_j^i(ζ) + ξ_j^i)")
    print("For actual usage, call generate_theoretical_user_preferences() method")
    print("with appropriate user, server, and system data.")