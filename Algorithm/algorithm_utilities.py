"""
Algorithm Module Utilities
Local utilities for the Algorithm module to ensure complete independence
"""

import numpy as np
import random
from typing import Dict, List, Tuple, Any


class AlgorithmUtilities:
    """
    Helper functions specifically for the Algorithm module
    Provides system generation, matching utilities, and calculations
    """
    
    @staticmethod
    def generate_users(config) -> List[Dict]:
        """Generate IoT users with random positions and characteristics"""
        users = []
        for i in range(config.num_users):
            # Random position within circular network area
            angle = random.uniform(0, 2 * np.pi)
            radius = random.uniform(0, config.network_area_size)
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            
            user = {
                'id': f'U{i+1}',
                'position': (x, y),
                'device_type': random.choice(['smartphone', 'tablet', 'iot_sensor']),
                'battery_level': random.uniform(0.3, 1.0),
                'computational_capability': random.uniform(1e9, 5e9)  # 1-5 GHz
            }
            users.append(user)
        
        return users
    
    @staticmethod
    def generate_servers(config) -> List[Dict]:
        """Generate fog servers with random positions and capabilities"""
        servers = []
        
        # Check if multi-level mode is enabled
        if hasattr(config, 'use_multilevel') and config.use_multilevel:
            server_id = 1
            
            # Level 1: Edge fog servers (closest to users)
            num_edge = getattr(config, 'edge_fog_servers', 2)
            for i in range(num_edge):
                angle = random.uniform(0, 2 * np.pi)
                radius = random.uniform(0, config.network_area_size * 0.4)
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
                
                server = {
                    'id': f'E{server_id}',
                    'level': 1,
                    'type': 'edge_fog',
                    'position': (x, y),
                    'computational_capability': random.uniform(1e9, 2e9),  # 1-2 GHz (was 2-4)
                    'available_resources': random.uniform(0.5, 1.0),
                    'energy_efficiency': random.uniform(0.7, 0.9),
                    'processing_cost': random.uniform(0.01, 0.03),
                    'communication_delay_base': random.uniform(0.001, 0.005)
                }
                servers.append(server)
                server_id += 1
            
            # Level 2: Regional fog servers (intermediate)
            num_regional = getattr(config, 'regional_fog_servers', 2)
            for i in range(num_regional):
                angle = random.uniform(0, 2 * np.pi)
                radius = random.uniform(config.network_area_size * 0.4, config.network_area_size * 0.7)
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
                
                server = {
                    'id': f'R{server_id}',
                    'level': 2,
                    'type': 'regional_fog',
                    'position': (x, y),
                    'computational_capability': random.uniform(2e9, 4e9),  # 2-4 GHz (was 4-8)
                    'available_resources': random.uniform(0.6, 1.0),
                    'energy_efficiency': random.uniform(0.6, 0.8),
                    'processing_cost': random.uniform(0.005, 0.015),
                    'communication_delay_base': random.uniform(0.005, 0.015)
                }
                servers.append(server)
                server_id += 1
            
            # Level 3: Cloud servers (most powerful, farthest)
            num_cloud = getattr(config, 'cloud_servers', 1)
            for i in range(num_cloud):
                angle = random.uniform(0, 2 * np.pi)
                radius = random.uniform(config.network_area_size * 0.7, config.network_area_size)
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
                
                server = {
                    'id': f'C{server_id}',
                    'level': 3,
                    'type': 'cloud',
                    'position': (x, y),
                    'computational_capability': random.uniform(4e9, 8e9),  # 4-8 GHz (was 8-16)
                    'available_resources': random.uniform(0.8, 1.0),
                    'energy_efficiency': random.uniform(0.5, 0.7),
                    'processing_cost': random.uniform(0.001, 0.008),
                    'communication_delay_base': random.uniform(0.02, 0.05)
                }
                servers.append(server)
                server_id += 1
        else:
            # Single-level mode: traditional fog servers
            for i in range(config.num_servers):
                angle = random.uniform(0, 2 * np.pi)
                radius = random.uniform(0, config.network_area_size)
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
                
                server = {
                    'id': f'S{i+1}',
                    'level': 1,
                    'type': 'fog',
                    'position': (x, y),
                    'computational_capability': random.uniform(1e9, 3e9),  # 1-3 GHz (was 2-5)
                    'available_resources': random.uniform(0.5, 1.0),
                    'energy_efficiency': random.uniform(0.6, 0.9),
                    'processing_cost': random.uniform(0.01, 0.05),
                    'communication_delay_base': 0.0
                }
                servers.append(server)
        
        return servers
    
    @staticmethod
    def generate_tasks(config, users: List[Dict], servers: List[Dict]) -> List[Dict]:
        """Generate tasks with different characteristics"""
        tasks = []
        
        # Check for fixed task count (for reproducibility)
        if hasattr(config, 'fixed_task_count') and config.fixed_task_count is not None:
            num_tasks = config.fixed_task_count
            print(f"Generating fixed number of tasks: {num_tasks}")
            
            # Distribute tasks among users
            tasks_per_user = [0] * len(users)
            for _ in range(num_tasks):
                user_idx = random.randint(0, len(users) - 1)
                tasks_per_user[user_idx] += 1
            
            print(f"Distributed tasks: min={min(tasks_per_user)}, max={max(tasks_per_user)}, avg={sum(tasks_per_user)/len(tasks_per_user):.1f} tasks/user")
        else:
            # Original behavior: tasks per user
            tasks_per_user = [config.tasks_per_user] * len(users)
            num_tasks = len(users) * config.tasks_per_user
            print(f"Generating tasks: {config.tasks_per_user} per user")
        
        # Check capacity model
        if config.use_hybrid_capacity:
            print(f"Note: Servers have HYBRID capacity (initial limit, then unlimited with waiting)")
        else:
            print(f"Note: Servers have UNLIMITED capacity (waiting time increases with load)")
        
        task_id = 1
        for user_idx, user in enumerate(users):
            for _ in range(tasks_per_user[user_idx]):
                task_type = random.randint(1, config.num_task_types)
                
                task = {
                    'id': f'T{task_id}',
                    'user_id': user['id'],
                    'type': task_type,
                    'task_type': random.randint(0, config.num_task_types - 1),
                    'computation_requirement': random.uniform(3000, 7000) * 1e6,  # CPU cycles (ULTRA load: 3000-7000M)
                    'data_size': random.uniform(0.5, 2.0) * 1e6,  # Data size in bits (hardcoded like Generation)
                    'deadline': random.uniform(1.0, 5.0),  # Deadline in seconds (hardcoded like Generation)
                    'priority': random.randint(1, 5),
                    'delay_tolerance': random.uniform(0.1, 1.0)  # Delay tolerance factor
                }
                tasks.append(task)
                task_id += 1
        
        capacity_msg = "hybrid capacity" if config.use_hybrid_capacity else "unlimited capacity"
        print(f"Generated {len(tasks)} tasks ({capacity_msg} - waiting time increases with load)")
        
        return tasks
    
    @staticmethod
    def calculate_euclidean_distance(pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two positions"""
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    @staticmethod
    def calculate_path_loss(distance: float, reference_distance: float, 
                           path_loss_exponent: float, reference_gain: float) -> float:
        """Calculate path loss using log-distance model"""
        if distance < reference_distance:
            distance = reference_distance
        return reference_gain * (reference_distance / distance) ** path_loss_exponent
    
    @staticmethod
    def calculate_shannon_capacity(bandwidth: float, snr: float) -> float:
        """Calculate Shannon channel capacity"""
        return bandwidth * np.log2(1 + snr)
    
    @staticmethod
    def calculate_transmission_delay(data_size: float, capacity: float) -> float:
        """Calculate transmission delay"""
        if capacity <= 0:
            return float('inf')
        return data_size / capacity
    
    @staticmethod
    def initialize_matching_state(task_ids: List[str], server_ids: List[str]) -> Tuple[Dict, Dict]:
        """
        Initialize the matching algorithm state
        
        Returns:
            Tuple of (server_assignments, task_current_preference)
        """
        server_assignments = {server_id: [] for server_id in server_ids}
        task_current_preference = {task_id: 0 for task_id in task_ids}
        
        return server_assignments, task_current_preference
    
    @staticmethod
    def sort_by_preference(items: List[Any], preference_function, reverse: bool = True) -> List[Any]:
        """
        Sort items by preference function
        
        Args:
            items: List of items to sort
            preference_function: Function that takes an item and returns a preference score
            reverse: Sort in descending order (higher preference first) if True
            
        Returns:
            Sorted list of items
        """
        return sorted(items, key=preference_function, reverse=reverse)


def set_random_seeds(seed: int):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)


def print_header(title: str, width: int = 70):
    """Print a formatted header"""
    print("\n" + "="*width)
    print(title.center(width))
    print("="*width)


def print_section(title: str, width: int = 70):
    """Print a formatted section header"""
    print("\n" + "="*width)
    print(title)
    print("="*width)


def format_time_duration(seconds: float) -> str:
    """Format time duration in a human-readable format"""
    if seconds < 1:
        return f"{seconds*1000:.2f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.2f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.2f}s"


def main():
    """Example usage of the Algorithm utilities"""
    print("=== Algorithm Module Utilities ===")
    print("This module provides local utilities for the Algorithm module.")
    print("\nKey Features:")
    print("  • System generation (users, servers, tasks)")
    print("  • Distance and channel calculations")
    print("  • Matching algorithm utilities")
    print("  • Formatting and helper functions")
    print("\nFor actual usage, import AlgorithmUtilities and other functions as needed.")


if __name__ == "__main__":
    main()
