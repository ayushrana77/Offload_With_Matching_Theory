"""
Utility Functions for Matching Theory Task Offloading Algorithm
Contains common utility functions used across different modules
"""

import numpy as np
import random
import math
from typing import Dict, List, Tuple, Optional, Any


class SystemUtilities:
    """Collection of system-wide utility functions"""
    
    @staticmethod
    def calculate_euclidean_distance(pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """
        Calculate Euclidean distance between two positions
        
        Args:
            pos1: First position (x1, y1)
            pos2: Second position (x2, y2)
            
        Returns:
            Euclidean distance between the positions
        """
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    @staticmethod
    def calculate_path_loss(distance: float, reference_distance: float = 1.0, 
                           path_loss_exponent: float = 2.0, reference_gain: float = 1e-3) -> float:
        """
        Calculate path loss for wireless communication
        
        Args:
            distance: Distance between transmitter and receiver
            reference_distance: Reference distance for path loss model
            path_loss_exponent: Path loss exponent (typically 2.0 for free space)
            reference_gain: Reference channel gain at reference distance
            
        Returns:
            Channel gain considering path loss
        """
        distance = max(distance, reference_distance)
        return reference_gain * (reference_distance / distance) ** path_loss_exponent
    
    @staticmethod
    def calculate_shannon_capacity(bandwidth: float, snr: float) -> float:
        """
        Calculate Shannon channel capacity
        
        Args:
            bandwidth: Channel bandwidth in Hz
            snr: Signal-to-noise ratio
            
        Returns:
            Channel capacity in bits/second
        """
        return bandwidth * math.log2(1 + snr)
    
    @staticmethod
    def calculate_transmission_delay(data_size: float, capacity: float) -> float:
        """
        Calculate transmission delay
        
        Args:
            data_size: Size of data to transmit (bits)
            capacity: Channel capacity (bits/second)
            
        Returns:
            Transmission delay in seconds
        """
        return data_size / capacity if capacity > 0 else float('inf')
    
    @staticmethod
    def calculate_computation_delay(computation_requirement: float, cpu_frequency: float) -> float:
        """
        Calculate computation delay
        
        Args:
            computation_requirement: Required CPU cycles
            cpu_frequency: CPU frequency in Hz
            
        Returns:
            Computation delay in seconds
        """
        return computation_requirement / cpu_frequency if cpu_frequency > 0 else float('inf')
    
    @staticmethod
    def calculate_energy_consumption(power: float, time: float) -> float:
        """
        Calculate energy consumption
        
        Args:
            power: Power consumption in Watts
            time: Time duration in seconds
            
        Returns:
            Energy consumption in Joules
        """
        return power * time
    
    @staticmethod
    def normalize_score(value: float, min_val: float, max_val: float) -> float:
        """
        Normalize a value to [0, 1] range
        
        Args:
            value: Value to normalize
            min_val: Minimum value in the range
            max_val: Maximum value in the range
            
        Returns:
            Normalized value between 0 and 1
        """
        if max_val == min_val:
            return 0.5  # Return middle value if no variation
        return (value - min_val) / (max_val - min_val)
    
    @staticmethod
    def weighted_score(scores: List[float], weights: List[float]) -> float:
        """
        Calculate weighted score from multiple criteria
        
        Args:
            scores: List of individual scores
            weights: List of weights for each score
            
        Returns:
            Weighted combined score
        """
        if len(scores) != len(weights):
            raise ValueError("Number of scores must match number of weights")
        
        total_weight = sum(weights)
        if total_weight == 0:
            return 0
        
        return sum(score * weight for score, weight in zip(scores, weights)) / total_weight

    @staticmethod
    def generate_users(config) -> List[Dict]:
        """
        Generate IoT users with random positions and characteristics
        
        Args:
            config: System configuration object
            
        Returns:
            List of user dictionaries
        """
        users = []
        for i in range(config.num_users):
            user = {
                'id': f'U{i+1}',
                'position': generate_random_position(config.network_area_size),
                'computational_capability': random.uniform(0.5, 1.5) * 1e9,  # GHz
                'energy_capacity': random.uniform(50, 100),  # Energy units
                'task_arrival_rate': random.uniform(0.1, 0.5),  # Tasks per second
                'mobility': random.uniform(0, 5)  # Mobility factor
            }
            users.append(user)
        return users
    
    @staticmethod
    def generate_servers(config) -> List[Dict]:
        """
        Generate fog servers with random positions and capabilities
        
        Args:
            config: System configuration object
            
        Returns:
            List of server dictionaries
        """
        servers = []
        
        if hasattr(config, 'use_multilevel') and config.use_multilevel:
            # Generate multi-level server hierarchy
            servers.extend(SystemUtilities.generate_multilevel_servers(config))
        else:
            # Generate single-level servers (original behavior)
            for i in range(config.num_servers):
                server = {
                    'id': f'S{i+1}',
                    'level': 1,
                    'type': 'fog_server',
                    'position': generate_random_position(config.network_area_size),
                    'computational_capability': random.uniform(2.0, 4.0) * 1e9,  # GHz
                    'available_resources': random.uniform(0.7, 1.0),  # Resource availability
                    'processing_cost': random.uniform(0.1, 0.3),  # Cost per computation
                    'energy_efficiency': random.uniform(0.5, 1.0),  # Energy efficiency factor
                    'initial_capacity': config.initial_server_capacity
                }
                servers.append(server)
        
        return servers
    
    @staticmethod
    def generate_multilevel_servers(config) -> List[Dict]:
        """
        Generate servers for multi-level hierarchy (edge → regional → cloud)
        
        Args:
            config: System configuration object
            
        Returns:
            List of server dictionaries across all hierarchy levels
        """
        servers = []
        
        # Level 1: Edge Fog Servers (closest to IoT devices)
        for i in range(config.edge_fog_servers):
            server = {
                'id': f'E{i+1}',
                'level': 1,
                'type': 'edge_fog',
                'position': generate_random_position(config.network_area_size),
                'computational_capability': random.uniform(*config.edge_fog_cpu_range),
                'initial_capacity': config.edge_fog_capacity,
                'available_resources': random.uniform(0.6, 0.8),
                'coverage_radius': 50.0,  # Limited coverage area
                'processing_cost': random.uniform(0.15, 0.25),
                'energy_efficiency': random.uniform(0.6, 0.8),
                'communication_delay_base': config.iot_to_edge_delay
            }
            servers.append(server)
        
        # Level 2: Regional Fog Servers (intermediate layer)
        for i in range(config.regional_fog_servers):
            server = {
                'id': f'R{i+1}',
                'level': 2,
                'type': 'regional_fog',
                'position': generate_random_position(config.network_area_size),
                'computational_capability': random.uniform(*config.regional_fog_cpu_range),
                'initial_capacity': config.regional_fog_capacity,
                'available_resources': random.uniform(0.8, 0.95),
                'coverage_radius': 200.0,  # Broader coverage area
                'processing_cost': random.uniform(0.08, 0.15),
                'energy_efficiency': random.uniform(0.8, 0.95),
                'communication_delay_base': config.edge_to_regional_delay
            }
            servers.append(server)
        
        # Level 3: Cloud Servers (highest layer)
        for i in range(config.cloud_servers):
            server = {
                'id': f'C{i+1}',
                'level': 3,
                'type': 'cloud',
                'position': (config.network_area_size/2, config.network_area_size/2),  # Central position
                'computational_capability': random.uniform(*config.cloud_cpu_range),
                'initial_capacity': config.cloud_capacity,
                'available_resources': random.uniform(0.9, 1.0),
                'coverage_radius': float('inf'),  # Global coverage
                'processing_cost': random.uniform(0.02, 0.08),
                'energy_efficiency': random.uniform(0.9, 1.0),
                'communication_delay_base': config.regional_to_cloud_delay
            }
            servers.append(server)
        
        print(f"Generated multi-level server hierarchy:")
        print(f"  Level 1 (Edge): {config.edge_fog_servers} servers")
        print(f"  Level 2 (Regional): {config.regional_fog_servers} servers")
        print(f"  Level 3 (Cloud): {config.cloud_servers} servers")
        
        return servers
    
    @staticmethod
    def generate_tasks(config, users, servers) -> List[Dict]:
        """
        Generate tasks with different characteristics
        
        Args:
            config: System configuration object
            users: List of user dictionaries
            servers: List of server dictionaries
            
        Returns:
            List of task dictionaries
        """
        tasks = []
        task_id = 1
        
        # With unlimited capacity model, no hard capacity limit
        # But we still need reasonable task generation
        
        # Determine number of tasks to generate
        if hasattr(config, 'fixed_task_count') and config.fixed_task_count is not None:
            # Use fixed task count
            num_tasks_to_generate = config.fixed_task_count
            print(f"Generating fixed number of tasks: {num_tasks_to_generate}")
            print(f"Note: Servers have UNLIMITED capacity (waiting time increases with load)")
            
            # Distribute tasks evenly among users
            tasks_per_user = num_tasks_to_generate // len(users)
            remaining_tasks = num_tasks_to_generate % len(users)
            
            for i, user in enumerate(users):
                # Each user gets base number + possibly 1 extra task
                user_task_count = tasks_per_user + (1 if i < remaining_tasks else 0)
                
                for _ in range(user_task_count):
                    task = {
                        'id': f'T{task_id}',
                        'user_id': user['id'],
                        'task_type': random.randint(0, config.num_task_types - 1),
                        'data_size': random.uniform(0.5, 2.0) * 1e6,  # Data size in bits
                        'computation_requirement': random.uniform(100, 1000) * 1e6,  # CPU cycles
                        'deadline': random.uniform(1.0, 5.0),  # Deadline in seconds
                        'priority': random.randint(1, 5),  # Priority level (1=low, 5=high)
                        'delay_tolerance': random.uniform(0.1, 1.0)  # Delay tolerance factor
                    }
                    tasks.append(task)
                    task_id += 1
                    
        else:
            # Generate tasks based on number of users (no capacity constraint)
            default_tasks_per_user = 2  # Default: 2 tasks per user
            tasks_per_user = default_tasks_per_user
            remaining_tasks = 0
            
            print(f"Generating tasks: {tasks_per_user} per user (unlimited capacity model)")
            
            for i, user in enumerate(users):
                # Each user gets default number of tasks
                num_tasks = tasks_per_user + (1 if i < remaining_tasks else 0)
                
                for _ in range(num_tasks):
                    task = {
                        'id': f'T{task_id}',
                        'user_id': user['id'],
                        'task_type': random.randint(0, config.num_task_types - 1),
                        'data_size': random.uniform(0.5, 2.0) * 1e6,  # Data size in bits
                        'computation_requirement': random.uniform(100, 1000) * 1e6,  # CPU cycles
                        'deadline': random.uniform(1.0, 5.0),  # Deadline in seconds
                        'priority': random.randint(1, 5),  # Priority level (1=low, 5=high)
                        'delay_tolerance': random.uniform(0.1, 1.0)  # Delay tolerance factor
                    }
                    tasks.append(task)
                    task_id += 1
        
        print(f"Generated {len(tasks)} tasks (unlimited capacity - waiting time increases with load)")
        return tasks


class PreferenceUtilities:
    """Utility functions for preference generation and manipulation"""
    
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
    
    @staticmethod
    def add_preference_randomization(preferences: List[str], randomization_probability: float = 0.2) -> List[str]:
        """
        Add randomization to preference lists to avoid identical preferences
        
        Args:
            preferences: Original preference list
            randomization_probability: Probability of swapping adjacent elements
            
        Returns:
            Randomized preference list
        """
        randomized_prefs = preferences.copy()
        for i in range(len(randomized_prefs) - 1):
            if random.random() < randomization_probability:
                randomized_prefs[i], randomized_prefs[i+1] = randomized_prefs[i+1], randomized_prefs[i]
        return randomized_prefs
    
    @staticmethod
    def validate_preference_list(preferences: List[str], valid_items: List[str]) -> bool:
        """
        Validate that preference list contains only valid items
        
        Args:
            preferences: Preference list to validate
            valid_items: List of valid items
            
        Returns:
            True if preference list is valid, False otherwise
        """
        return all(item in valid_items for item in preferences)
    
    @staticmethod
    def generate_random_preference_list(items: List[str]) -> List[str]:
        """
        Generate a random preference list from items
        
        Args:
            items: List of items to create preferences from
            
        Returns:
            Randomly shuffled preference list
        """
        preferences = items.copy()
        random.shuffle(preferences)
        return preferences


class MatchingUtilities:
    """Utility functions for matching algorithms"""
    
    @staticmethod
    def initialize_matching_state(agents: List[str], partners: List[str]) -> Tuple[Dict[str, List[str]], Dict[str, int]]:
        """
        Initialize matching algorithm state
        
        Args:
            agents: List of agents (proposers)
            partners: List of partners (receivers)
            
        Returns:
            Tuple of (assignments dictionary, current preference indices)
        """
        assignments = {partner: [] for partner in partners}
        current_preferences = {agent: 0 for agent in agents}
        return assignments, current_preferences
    
    @staticmethod
    def is_matching_stable(assignments: Dict[str, List[str]], 
                          agent_preferences: Dict[str, List[str]], 
                          partner_preferences: Dict[str, List[str]]) -> bool:
        """
        Check if a matching is stable (simplified check)
        
        Args:
            assignments: Current assignments
            agent_preferences: Agent preference lists
            partner_preferences: Partner preference lists
            
        Returns:
            True if matching appears stable, False otherwise
        """
        # Simplified stability check - can be enhanced
        for partner, assigned_agents in assignments.items():
            if len(assigned_agents) > 0:
                # Check if partner prefers assigned agents over others
                partner_prefs = partner_preferences.get(partner, [])
                for agent in assigned_agents:
                    if agent not in partner_prefs:
                        return False
        return True
    
    @staticmethod
    def calculate_matching_statistics(assignments: Dict[str, List[str]], 
                                    capacities: Dict[str, int]) -> Dict[str, Any]:
        """
        Calculate statistics for a matching result
        
        Args:
            assignments: Assignment dictionary
            capacities: Capacity constraints
            
        Returns:
            Dictionary of matching statistics
        """
        total_assigned = sum(len(assigned) for assigned in assignments.values())
        
        # Handle unlimited capacity (inf) vs limited capacity
        has_unlimited = any(cap == float('inf') for cap in capacities.values())
        
        if has_unlimited:
            # For unlimited capacity: report task distribution instead of utilization
            total_capacity = float('inf')
            utilization = {}
            for partner, capacity in capacities.items():
                assigned_count = len(assignments.get(partner, []))
                # For unlimited capacity, "utilization" is the task count (not a ratio)
                utilization[partner] = assigned_count
            
            # Average utilization = average task count per server
            avg_utilization = total_assigned / len(capacities) if capacities else 0
        else:
            # Limited capacity: traditional calculation
            total_capacity = sum(capacities.values())
            utilization = {}
            for partner, capacity in capacities.items():
                assigned_count = len(assignments.get(partner, []))
                utilization[partner] = assigned_count / capacity if capacity > 0 else 0
            
            avg_utilization = sum(utilization.values()) / len(utilization) if utilization else 0
        
        return {
            'total_assigned': total_assigned,
            'total_capacity': total_capacity,
            'utilization': utilization,
            'average_utilization': avg_utilization,
            'capacity_efficiency': total_assigned / total_capacity if total_capacity > 0 and total_capacity != float('inf') else 0,
            'unlimited_capacity_mode': has_unlimited
        }


class MetricsUtilities:
    """Utility functions for performance metrics calculation"""
    
    @staticmethod
    def calculate_completion_time_metrics(completion_times: List[float]) -> Dict[str, float]:
        """
        Calculate completion time statistics
        
        Args:
            completion_times: List of task completion times
            
        Returns:
            Dictionary of completion time metrics
        """
        if not completion_times:
            return {
                'max_completion_time': 0,
                'avg_completion_time': 0,
                'min_completion_time': 0,
                'total_completion_time': 0,
                'std_completion_time': 0
            }
        
        return {
            'max_completion_time': max(completion_times),
            'avg_completion_time': np.mean(completion_times),
            'min_completion_time': min(completion_times),
            'total_completion_time': sum(completion_times),
            'std_completion_time': np.std(completion_times)
        }
    
    @staticmethod
    def calculate_system_efficiency(assignment_success_rate: float, 
                                  average_utilization: float,
                                  energy_per_task: float,
                                  max_energy_per_task: float = 1.0) -> float:
        """
        Calculate overall system efficiency score
        
        Args:
            assignment_success_rate: Fraction of successfully assigned tasks
            average_utilization: Average server utilization
            energy_per_task: Energy consumption per task
            max_energy_per_task: Maximum acceptable energy per task
            
        Returns:
            System efficiency score (0-1)
        """
        energy_efficiency = 1 - min(energy_per_task / max_energy_per_task, 1.0)
        return (assignment_success_rate + average_utilization + energy_efficiency) / 3
    
    @staticmethod
    def print_performance_summary(metrics: Dict[str, Any], title: str = "Performance Summary"):
        """
        Print formatted performance summary
        
        Args:
            metrics: Dictionary of performance metrics
            title: Title for the summary
        """
        print(f"\n{'='*60}")
        print(f"{title.center(60)}")
        print("="*60)
        
        # Format and print key metrics
        if 'max_completion_time' in metrics:
            print(f"Max Completion Time:     {metrics['max_completion_time']:.4f}s")
        if 'avg_completion_time' in metrics:
            print(f"Avg Completion Time:     {metrics['avg_completion_time']:.4f}s")
        if 'total_delay' in metrics:
            print(f"Total System Delay:      {metrics['total_delay']:.4f}s")
        if 'total_cost' in metrics:
            print(f"Total Cost:              {metrics['total_cost']:.2f} units")
        if 'total_energy' in metrics:
            print(f"Total Energy:            {metrics['total_energy']:.4f} J")
        if 'assignment_success_rate' in metrics:
            print(f"Assignment Success Rate: {metrics['assignment_success_rate']:.1%}")
        if 'average_utilization' in metrics:
            print(f"Average Utilization:     {metrics.get('average_utilization', 0):.1%}")
        
        print("="*60)


class ValidationUtilities:
    """Utility functions for data validation"""
    

    
    @staticmethod
    def validate_allocation_matrix(allocation_matrix: np.ndarray) -> bool:
        """
        Validate allocation matrix constraints
        
        Args:
            allocation_matrix: n x m allocation matrix
            
        Returns:
            True if allocation matrix is valid, False otherwise
        """
        if allocation_matrix is None:
            return False
        
        # Check if matrix contains only 0s and 1s
        if not np.all((allocation_matrix == 0) | (allocation_matrix == 1)):
            return False
        
        # Check if each task is assigned to exactly one server
        n, m = allocation_matrix.shape
        for j in range(m):  # For each task
            if not np.isclose(np.sum(allocation_matrix[:, j]), 1.0):
                return False
        
        return True
    
    @staticmethod
    def validate_preferences(preferences: Dict[str, List[str]], 
                           valid_agents: List[str], 
                           valid_partners: List[str]) -> List[str]:
        """
        Validate preference dictionaries
        
        Args:
            preferences: Preference dictionary to validate
            valid_agents: List of valid agents
            valid_partners: List of valid partners
            
        Returns:
            List of validation error messages
        """
        errors = []
        
        # Check if all agents have preferences
        for agent in valid_agents:
            if agent not in preferences:
                errors.append(f"Agent {agent} has no preferences defined")
            else:
                # Check if all preferences are valid partners
                for partner in preferences[agent]:
                    if partner not in valid_partners:
                        errors.append(f"Agent {agent} has invalid preference: {partner}")
        
        return errors


def set_random_seeds(seed: int = 42):
    """
    Set random seeds for reproducible results
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)


def generate_random_position(area_size: float) -> Tuple[float, float]:
    """
    Generate random position within specified area
    
    Args:
        area_size: Size of the square area
        
    Returns:
        Random position (x, y) within the area
    """
    return (random.uniform(0, area_size), random.uniform(0, area_size))


def format_time_duration(seconds: float) -> str:
    """
    Format time duration in a readable format
    
    Args:
        seconds: Time duration in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 0.001:
        return f"{seconds*1000000:.1f}μs"
    elif seconds < 1:
        return f"{seconds*1000:.1f}ms"
    elif seconds < 60:
        return f"{seconds:.3f}s"
    else:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"


def print_header(title: str, width: int = 80, char: str = "="):
    """
    Print formatted header
    
    Args:
        title: Header title
        width: Total width of header
        char: Character to use for header decoration
    """
    print(char * width)
    print(title.center(width))
    print(char * width)


def print_section(title: str, width: int = 60, char: str = "-"):
    """
    Print formatted section header
    
    Args:
        title: Section title
        width: Total width of section header
        char: Character to use for decoration
    """
    print(f"\n{char * width}")
    print(f"{title.upper()}")
    print(char * width)