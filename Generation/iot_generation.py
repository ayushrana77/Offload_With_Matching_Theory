"""
IoT User and Task Generation Module
Handles generation of IoT users/devices and their associated tasks
"""

import random
import numpy as np
from typing import Dict, List, Tuple


class IoTGenerator:
    """
    Generator for IoT users/devices and their tasks
    Separated from server generation for better modularity
    """
    
    def __init__(self, config):
        """
        Initialize IoT generator
        
        Args:
            config: SystemConfiguration instance
        """
        self.config = config
    
    def generate_iot_users(self) -> List[Dict]:
        """
        Generate IoT users/devices with positions and characteristics
        
        Returns:
            List of user dictionaries with id, position, device type, etc.
        """
        users = []
        
        for i in range(self.config.num_users):
            user = {
                'id': f'U{i+1}',
                'position': self._generate_random_position(),
                'mobility': random.choice(['static', 'mobile']),
                'device_type': random.choice(['sensor', 'smartphone', 'camera', 'wearable', 'smart_home']),
                'battery_level': random.uniform(0.3, 1.0),  # 30% - 100%
                'processing_capability': random.uniform(1e8, 5e8),  # 0.1-0.5 GHz
                'energy_budget': random.uniform(0.1, 1.0)  # Joules
            }
            users.append(user)
        
        print(f"✅ Generated {len(users)} IoT users/devices")
        return users
    
    def generate_tasks(self, users: List[Dict], servers: List[Dict]) -> List[Dict]:
        """
        Generate computation tasks for IoT users
        
        Args:
            users: List of user dictionaries
            servers: List of server dictionaries (for context)
            
        Returns:
            List of task dictionaries with computation requirements
        """
        tasks = []
        
        # Determine number of tasks
        if self.config.fixed_task_count:
            num_tasks = self.config.fixed_task_count
            print(f"Generating fixed number of tasks: {num_tasks}")
        else:
            num_tasks = self.config.num_users * random.randint(5, 15)
            print(f"Generating {5}-{15} tasks per user")
        
        # Get task type definitions
        task_types = self._get_task_type_definitions()
        
        # Track tasks per user for balanced distribution
        tasks_per_user = {user['id']: 0 for user in users}
        
        for i in range(num_tasks):
            # Select user (prefer balanced distribution)
            if self.config.fixed_task_count:
                # More balanced distribution for fixed count
                min_tasks = min(tasks_per_user.values())
                candidate_users = [uid for uid, count in tasks_per_user.items() if count == min_tasks]
                user_id = random.choice(candidate_users)
                user = next(u for u in users if u['id'] == user_id)
            else:
                user = random.choice(users)
            
            # Select task type based on device type
            task_type = self._select_task_type_for_device(user['device_type'], task_types)
            
            # Create task
            task = {
                'id': f'T{i+1}',
                'user_id': user['id'],
                'type': task_type['name'],
                'computation_requirement': random.uniform(
                    task_type['comp_min'],
                    task_type['comp_max']
                ),  # CPU cycles
                'data_size': random.uniform(
                    task_type['data_min'],
                    task_type['data_max']
                ),  # Bytes
                'priority': random.randint(
                    self.config.min_task_priority,
                    self.config.max_task_priority
                ),
                'deadline': random.uniform(
                    task_type['deadline_min'],
                    task_type['deadline_max']
                ),  # seconds
                'arrival_time': random.uniform(0, 10),  # Arrival time in simulation
                'energy_consumption_weight': task_type.get('energy_weight', 1.0),
                'urgency': task_type.get('urgency', 'normal')
            }
            tasks.append(task)
            tasks_per_user[user['id']] += 1
        
        # Print distribution statistics
        task_counts = list(tasks_per_user.values())
        print(f"Distributed tasks: min={min(task_counts)}, max={max(task_counts)}, avg={np.mean(task_counts):.1f} tasks/user")
        
        # Print capacity warning
        if self.config.use_hybrid_capacity:
            print(f"Note: Servers have HYBRID capacity (limited → unlimited when full)")
        else:
            print(f"Note: Servers have UNLIMITED capacity (waiting time increases with load)")
        
        print(f"Generated {len(tasks)} tasks ({'unlimited capacity - waiting time increases with load'})")
        
        return tasks
    
    def _generate_random_position(self) -> Tuple[float, float]:
        """Generate random position within network area"""
        return (
            random.uniform(0, self.config.network_area_size),
            random.uniform(0, self.config.network_area_size)
        )
    
    def _get_task_type_definitions(self) -> List[Dict]:
        """
        Define different task types with varying computational requirements
        
        Returns:
            List of task type definitions
        """
        return [
            {
                'name': 'video_processing',
                'comp_min': 5e9, 'comp_max': 10e9,  # High computation
                'data_min': 5e6, 'data_max': 20e6,  # 5-20 MB
                'deadline_min': 2.0, 'deadline_max': 5.0,
                'energy_weight': 1.5,
                'urgency': 'normal'
            },
            {
                'name': 'image_recognition',
                'comp_min': 2e9, 'comp_max': 5e9,  # Medium computation
                'data_min': 1e6, 'data_max': 5e6,  # 1-5 MB
                'deadline_min': 1.0, 'deadline_max': 3.0,
                'energy_weight': 1.2,
                'urgency': 'normal'
            },
            {
                'name': 'sensor_data',
                'comp_min': 1e8, 'comp_max': 5e8,  # Low computation
                'data_min': 1e4, 'data_max': 1e5,  # 10-100 KB
                'deadline_min': 0.5, 'deadline_max': 2.0,
                'energy_weight': 0.8,
                'urgency': 'low'
            },
            {
                'name': 'ar_rendering',
                'comp_min': 8e9, 'comp_max': 15e9,  # Very high computation
                'data_min': 10e6, 'data_max': 50e6,  # 10-50 MB
                'deadline_min': 0.1, 'deadline_max': 0.5,  # Strict deadline
                'energy_weight': 2.0,
                'urgency': 'high'
            },
            {
                'name': 'data_analytics',
                'comp_min': 3e9, 'comp_max': 7e9,  # Medium-high computation
                'data_min': 2e6, 'data_max': 10e6,  # 2-10 MB
                'deadline_min': 3.0, 'deadline_max': 10.0,
                'energy_weight': 1.3,
                'urgency': 'normal'
            },
            {
                'name': 'voice_processing',
                'comp_min': 5e8, 'comp_max': 2e9,  # Low-medium computation
                'data_min': 5e5, 'data_max': 2e6,  # 0.5-2 MB
                'deadline_min': 0.2, 'deadline_max': 1.0,
                'energy_weight': 0.9,
                'urgency': 'high'
            },
            {
                'name': 'face_detection',
                'comp_min': 3e9, 'comp_max': 6e9,  # Medium computation
                'data_min': 2e6, 'data_max': 8e6,  # 2-8 MB
                'deadline_min': 0.5, 'deadline_max': 2.0,
                'energy_weight': 1.4,
                'urgency': 'normal'
            },
            {
                'name': 'iot_control',
                'comp_min': 1e8, 'comp_max': 3e8,  # Very low computation
                'data_min': 5e3, 'data_max': 5e4,  # 5-50 KB
                'deadline_min': 0.1, 'deadline_max': 0.5,
                'energy_weight': 0.5,
                'urgency': 'critical'
            }
        ]
    
    def _select_task_type_for_device(self, device_type: str, task_types: List[Dict]) -> Dict:
        """
        Select appropriate task type based on device type
        
        Args:
            device_type: Type of IoT device
            task_types: Available task types
            
        Returns:
            Selected task type definition
        """
        # Device-specific task preferences
        device_task_mapping = {
            'sensor': ['sensor_data', 'iot_control'],
            'smartphone': ['image_recognition', 'video_processing', 'ar_rendering', 'face_detection'],
            'camera': ['video_processing', 'image_recognition', 'face_detection'],
            'wearable': ['sensor_data', 'voice_processing', 'iot_control'],
            'smart_home': ['iot_control', 'sensor_data', 'voice_processing']
        }
        
        # Get preferred task types for this device
        preferred_tasks = device_task_mapping.get(device_type, [])
        
        if preferred_tasks and random.random() < 0.7:  # 70% chance to use device-specific task
            task_name = random.choice(preferred_tasks)
            task_type = next((t for t in task_types if t['name'] == task_name), random.choice(task_types))
        else:
            task_type = random.choice(task_types)
        
        return task_type
    
    def get_user_summary(self, users: List[Dict]) -> Dict:
        """
        Get summary statistics for generated users
        
        Args:
            users: List of user dictionaries
            
        Returns:
            Dictionary with summary statistics
        """
        device_types = {}
        mobility_types = {}
        
        for user in users:
            device_type = user['device_type']
            mobility = user['mobility']
            
            device_types[device_type] = device_types.get(device_type, 0) + 1
            mobility_types[mobility] = mobility_types.get(mobility, 0) + 1
        
        return {
            'total_users': len(users),
            'device_types': device_types,
            'mobility_types': mobility_types,
            'avg_battery': np.mean([u['battery_level'] for u in users]),
            'avg_processing': np.mean([u['processing_capability'] for u in users])
        }
    
    def get_task_summary(self, tasks: List[Dict]) -> Dict:
        """
        Get summary statistics for generated tasks
        
        Args:
            tasks: List of task dictionaries
            
        Returns:
            Dictionary with summary statistics
        """
        task_types = {}
        urgency_levels = {}
        
        for task in tasks:
            task_type = task['type']
            urgency = task['urgency']
            
            task_types[task_type] = task_types.get(task_type, 0) + 1
            urgency_levels[urgency] = urgency_levels.get(urgency, 0) + 1
        
        return {
            'total_tasks': len(tasks),
            'task_types': task_types,
            'urgency_levels': urgency_levels,
            'avg_computation': np.mean([t['computation_requirement'] for t in tasks]),
            'avg_data_size': np.mean([t['data_size'] for t in tasks]),
            'avg_priority': np.mean([t['priority'] for t in tasks])
        }


def main():
    """Test IoT generation module"""
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    
    from config import SystemConfiguration
    
    config = SystemConfiguration(
        num_users=10,
        num_servers=5,
        fixed_task_count=50,
        random_seed=42
    )
    
    random.seed(42)
    np.random.seed(42)
    
    print("="*70)
    print("Testing IoT Generation Module")
    print("="*70)
    
    # Generate IoT users
    iot_gen = IoTGenerator(config)
    users = iot_gen.generate_iot_users()
    
    # Print user summary
    user_summary = iot_gen.get_user_summary(users)
    print(f"\n📱 User Summary:")
    print(f"  Total users: {user_summary['total_users']}")
    print(f"  Device types: {user_summary['device_types']}")
    print(f"  Mobility: {user_summary['mobility_types']}")
    
    # Generate tasks (need dummy servers)
    dummy_servers = [{'id': f'S{i+1}'} for i in range(config.num_servers)]
    tasks = iot_gen.generate_tasks(users, dummy_servers)
    
    # Print task summary
    task_summary = iot_gen.get_task_summary(tasks)
    print(f"\n📋 Task Summary:")
    print(f"  Total tasks: {task_summary['total_tasks']}")
    print(f"  Task types: {task_summary['task_types']}")
    print(f"  Urgency levels: {task_summary['urgency_levels']}")
    print(f"  Avg computation: {task_summary['avg_computation']/1e9:.2f} GHz cycles")
    print(f"  Avg data size: {task_summary['avg_data_size']/1e6:.2f} MB")
    
    print("\n✅ IoT Generation Module Test Complete")


if __name__ == "__main__":
    main()
