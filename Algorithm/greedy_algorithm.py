"""
Greedy Algorithm Implementation (Distance-Based)
Simple greedy approach: Always assign tasks to the nearest available server
For comparison with the Matching Theory approach
"""

import numpy as np
import random
import time
from typing import Dict, List

# Import from root config and other modules
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from config import SystemConfiguration
from Simulation.simulation_metrics import SimulationMetrics

# Use local algorithm utilities (no external module dependencies)
from .algorithm_utilities import (
    AlgorithmUtilities, set_random_seeds,
    print_header, print_section, format_time_duration
)


class GreedyTaskOffloadingAlgorithm:
    """
    Greedy algorithm: Always assign tasks to geographically nearest server
    No preference matching, no optimization - just simple distance-based assignment
    """
    
    def __init__(self, config: SystemConfiguration = None):
        """Initialize the greedy algorithm"""
        self.config = config or SystemConfiguration()
        
        # System components (same as matching theory)
        self.users = []
        self.servers = []
        self.tasks = []
        
        # System state
        self.distance_matrix = None
        self.channel_gains = None
        self.transmission_delays = None
        self.computation_costs = None
        self.server_capacities = {}
        self.server_waiting_times = {}
        
        # Algorithm results
        self.final_allocation = {}
        
        # Unified simulation and metrics calculator
        self.unified_calculator = None
        
    def initialize_system(self):
        """Initialize the system (same as matching theory algorithm)"""
        print("=== Initializing Greedy Algorithm System ===")
        
        # Validate configuration
        config_errors = self.config.validate()
        if config_errors:
            print("Configuration validation errors:")
            for error in config_errors:
                print(f"  - {error}")
            raise ValueError("Invalid system configuration")
        
        # Generate system components
        self.users = AlgorithmUtilities.generate_users(self.config)
        self.servers = AlgorithmUtilities.generate_servers(self.config)
        self.tasks = AlgorithmUtilities.generate_tasks(self.config, self.users, self.servers)
        
        # Calculate system matrices
        self._calculate_system_matrices()
        
        # Initialize server capacities
        self._initialize_server_capacities()
        
        print(f"System initialized: {len(self.users)} users, {len(self.servers)} servers, {len(self.tasks)} tasks")
    
    def _calculate_system_matrices(self):
        """Calculate distance, channel gains, transmission delays, and costs"""
        num_users = len(self.users)
        num_servers = len(self.servers)
        
        # Initialize matrices
        self.distance_matrix = np.zeros((num_users, num_servers))
        self.channel_gains = np.zeros((num_users, num_servers))
        self.transmission_delays = np.zeros((num_users, num_servers))
        self.computation_costs = np.zeros((num_users, num_servers))
        
        # Calculate for each user-server pair
        for i, user in enumerate(self.users):
            for j, server in enumerate(self.servers):
                # Calculate distance
                distance = AlgorithmUtilities.calculate_euclidean_distance(
                    user['position'], server['position']
                )
                self.distance_matrix[i][j] = distance
                
                # Calculate channel gain
                channel_gain = AlgorithmUtilities.calculate_path_loss(
                    distance, 
                    self.config.reference_distance,
                    self.config.path_loss_exponent, 
                    self.config.reference_gain
                )
                self.channel_gains[i][j] = channel_gain
                
                # Calculate transmission delay
                snr = (self.config.transmission_power * channel_gain) / self.config.noise_power
                capacity = AlgorithmUtilities.calculate_shannon_capacity(self.config.channel_bandwidth, snr)
                
                avg_task_size = 1e6  # 1 MB average
                base_transmission_delay = AlgorithmUtilities.calculate_transmission_delay(avg_task_size, capacity)
                
                # Add multi-level delays if enabled
                if hasattr(self.config, 'use_multilevel') and self.config.use_multilevel:
                    server_level = server.get('level', 1)
                    level_delay = server.get('communication_delay_base', 0.0)
                    total_transmission_delay = base_transmission_delay + level_delay
                else:
                    total_transmission_delay = base_transmission_delay
                
                self.transmission_delays[i][j] = total_transmission_delay
                
                # Calculate computation cost
                base_cost = server['processing_cost']
                resource_factor = 1.0 / server['available_resources']
                
                # Multi-level cost adjustments
                if hasattr(self.config, 'use_multilevel') and self.config.use_multilevel:
                    server_level = server.get('level', 1)
                    if server_level == 1:
                        level_multiplier = getattr(self.config, 'edge_cost_multiplier', 1.2)
                    elif server_level == 2:
                        level_multiplier = getattr(self.config, 'regional_cost_multiplier', 1.0)
                    else:
                        level_multiplier = getattr(self.config, 'cloud_cost_multiplier', 0.8)
                    
                    self.computation_costs[i][j] = base_cost * resource_factor * level_multiplier
                else:
                    self.computation_costs[i][j] = base_cost * resource_factor
    
    def _initialize_server_capacities(self):
        """Initialize server capacities (same hybrid model as matching theory)"""
        for server in self.servers:
            if self.config.use_hybrid_capacity:
                if hasattr(self.config, 'use_multilevel') and self.config.use_multilevel:
                    level = server.get('level', 1)
                    if level == 1:
                        initial_capacity = getattr(self.config, 'edge_fog_capacity', self.config.initial_server_capacity)
                    elif level == 2:
                        initial_capacity = getattr(self.config, 'regional_fog_capacity', self.config.initial_server_capacity * 2)
                    else:
                        initial_capacity = getattr(self.config, 'cloud_capacity', self.config.initial_server_capacity * 5)
                else:
                    initial_capacity = self.config.initial_server_capacity
                
                self.server_capacities[server['id']] = initial_capacity
                print(f"  {server['id']}: Initial capacity = {initial_capacity} tasks")
            else:
                self.server_capacities[server['id']] = float('inf')
            
            self.server_waiting_times[server['id']] = 0.0
        
        if self.config.use_hybrid_capacity:
            print(f"✅ Initialized servers with HYBRID capacity model")
        else:
            print(f"✅ Initialized servers with UNLIMITED capacity model")
    
    def run_greedy_algorithm(self) -> Dict[str, List[str]]:
        """
        Run the GREEDY algorithm - always assign to nearest server
        No preference matching, no optimization, just pure distance-based assignment
        """
        print("\n" + "="*80)
        print("🎯 RUNNING GREEDY ALGORITHM (DISTANCE-BASED)")
        print("="*80)
        print("Strategy: Always assign tasks to the geographically nearest server")
        print("No preference optimization, no load balancing consideration")
        
        # Initialize server assignments
        server_assignments = {server['id']: [] for server in self.servers}
        
        # Get task-to-user mapping
        task_to_user = {task['id']: task['user_id'] for task in self.tasks}
        
        print(f"\nProcessing {len(self.tasks)} tasks...")
        
        # Process each task in order
        for task_idx, task in enumerate(self.tasks):
            task_id = task['id']
            user_id = task['user_id']
            
            # Find user index
            user_idx = next(i for i, u in enumerate(self.users) if u['id'] == user_id)
            
            # Find nearest server based on distance only
            distances_to_servers = []
            for server_idx, server in enumerate(self.servers):
                distance = self.distance_matrix[user_idx][server_idx]
                distances_to_servers.append((distance, server['id']))
            
            # Sort by distance (nearest first)
            distances_to_servers.sort(key=lambda x: x[0])
            
            # Assign to nearest server (greedy choice)
            nearest_server_id = distances_to_servers[0][1]
            nearest_distance = distances_to_servers[0][0]
            
            # Add task to server
            server_assignments[nearest_server_id].append(task_id)
            
            if (task_idx + 1) % 500 == 0:
                print(f"  Processed {task_idx + 1}/{len(self.tasks)} tasks...")
            
            # Show first 10 assignments for debugging
            if task_idx < 10:
                print(f"  Task {task_id} (User {user_id}) → {nearest_server_id} (distance: {nearest_distance:.2f}m)")
        
        # Update waiting times based on final assignments
        self._update_server_waiting_times(server_assignments)
        
        # Print allocation summary
        print("\n" + "="*70)
        print("✅ GREEDY ALLOCATION COMPLETE")
        print("="*70)
        
        # Show final allocation
        print(f"\n🏆 FINAL TASK ALLOCATION (GREEDY DISTANCE-BASED):")
        
        # Show statistics by level if multi-level
        if hasattr(self.config, 'use_multilevel') and self.config.use_multilevel:
            # Group servers by level
            servers_by_level = {1: [], 2: [], 3: []}
            for server in self.servers:
                level = server.get('level', 1)
                servers_by_level[level].append(server)
            
            for level in sorted(servers_by_level.keys()):
                level_servers = servers_by_level[level]
                if level_servers:
                    level_name = {1: "EDGE", 2: "REGIONAL", 3: "CLOUD"}[level]
                    print(f"  🏗️  LEVEL {level} ({level_name}) SERVERS:")
                    
                    for server in level_servers:
                        server_id = server['id']
                        assigned_tasks = server_assignments.get(server_id, [])
                        num_tasks = len(assigned_tasks)
                        waiting_time = self.server_waiting_times.get(server_id, 0.0)
                        capacity = self.server_capacities.get(server_id, 0)
                        
                        if num_tasks > 0:
                            over_capacity = " ⚠️ OVERLOADED" if num_tasks > capacity else ""
                            print(f"    {server_id}: {num_tasks} tasks (capacity: {capacity}, waiting: {waiting_time:.3f}s){over_capacity}")
                        else:
                            print(f"    {server_id}: 0 tasks (IDLE)")
        else:
            # Single-level display
            for server_id, assigned_tasks in server_assignments.items():
                num_tasks = len(assigned_tasks)
                waiting_time = self.server_waiting_times.get(server_id, 0.0)
                capacity = self.server_capacities.get(server_id, 0)
                
                if num_tasks > 0:
                    over_capacity = " ⚠️ OVERLOADED" if num_tasks > capacity else ""
                    print(f"  {server_id}: {num_tasks} tasks (capacity: {capacity}, waiting: {waiting_time:.3f}s){over_capacity}")
                else:
                    print(f"  {server_id}: 0 tasks (IDLE)")
        
        # Calculate and show imbalance
        task_counts = [len(tasks) for tasks in server_assignments.values()]
        max_tasks = max(task_counts)
        min_tasks = min(task_counts)
        avg_tasks = sum(task_counts) / len(task_counts)
        
        print(f"\n📊 LOAD DISTRIBUTION:")
        print(f"  Max tasks on one server: {max_tasks}")
        print(f"  Min tasks on one server: {min_tasks}")
        print(f"  Average tasks per server: {avg_tasks:.1f}")
        print(f"  Imbalance ratio: {max_tasks/max(1, min_tasks):.2f}x")
        
        self.final_allocation = server_assignments
        return server_assignments
    
    def _update_server_waiting_times(self, server_assignments: Dict[str, List[str]]):
        """Update server waiting times based on assignments"""
        for server_id in self.server_capacities.keys():
            assigned_tasks = server_assignments.get(server_id, [])
            num_assigned = len(assigned_tasks)
            
            server_info = next(s for s in self.servers if s['id'] == server_id)
            
            is_hybrid = self.config.use_hybrid_capacity
            initial_capacity = self.server_capacities.get(server_id, 1)
            
            # Calculate total processing time
            total_processing_time = 0.0
            for task_id in assigned_tasks:
                task_info = next(t for t in self.tasks if t['id'] == task_id)
                processing_time = task_info['computation_requirement'] / server_info['computational_capability']
                total_processing_time += processing_time
            
            if is_hybrid and num_assigned <= initial_capacity:
                # Under capacity: parallel processing
                waiting_time = total_processing_time / min(num_assigned, initial_capacity) if num_assigned > 0 else 0.0
            else:
                # Over capacity: sequential with penalty
                tasks_over_capacity = max(0, num_assigned - initial_capacity) if is_hybrid else num_assigned
                base_waiting_time = total_processing_time
                
                if is_hybrid:
                    queue_penalty = (tasks_over_capacity ** self.config.waiting_time_penalty_exponent) * self.config.waiting_time_increment
                else:
                    queue_penalty = (num_assigned ** self.config.waiting_time_penalty_exponent) * self.config.waiting_time_increment
                
                waiting_time = base_waiting_time + queue_penalty
            
            self.server_waiting_times[server_id] = waiting_time
    
    def calculate_performance_metrics(self, allocation: Dict[str, List[str]]) -> Dict:
        """Calculate performance metrics using simulation"""
        if self.unified_calculator is None:
            self.unified_calculator = SimulationMetrics(
                self.tasks, self.servers, self.users, self.server_capacities, 
                self.transmission_delays, self.config
            )
        
        unified_results = self.unified_calculator.run_simulation_and_calculate_metrics(allocation)
        return unified_results['numerical_results']
    
    def simulate_task_execution(self, allocation: Dict[str, List[str]]) -> Dict:
        """Simulate task execution"""
        if self.unified_calculator is None:
            self.unified_calculator = SimulationMetrics(
                self.tasks, self.servers, self.users, self.server_capacities, 
                self.transmission_delays, self.config
            )
        
        unified_results = self.unified_calculator.run_simulation_and_calculate_metrics(allocation)
        return unified_results['simulation_data']
    
    def run_complete_algorithm(self):
        """Run the complete greedy algorithm with analysis"""
        print_header("GREEDY ALGORITHM FOR TASK OFFLOADING (DISTANCE-BASED)", 90)
        
        start_time = time.time()
        
        # Initialize system
        self.initialize_system()
        
        # Run greedy algorithm (no preference calculation needed)
        allocation = self.run_greedy_algorithm()
        
        # Simulate task execution
        print("\n" + "="*80)
        print("🔄 SIMULATING TASK EXECUTION")
        print("="*80)
        simulation_results = self.simulate_task_execution(allocation)
        
        # Calculate performance metrics
        numerical_results = self.calculate_performance_metrics(allocation)
        
        # Print results
        print_section("Numerical Results Summary")
        print("Performance Metrics:")
        print(f"  T̄_M (Worst completion time): {numerical_results['worst_completion_time_TM']:.4f}s")
        print(f"  T̄_T (Mean completion time): {numerical_results['mean_completion_time_TT']:.4f}s")
        print(f"  T̄_W (Mean waiting time): {numerical_results['mean_waiting_time_TW']:.4f}s")
        print(f"  I_J (Jain's fairness index): {numerical_results['jains_index_IJ']:.4f}")
        
        execution_time = time.time() - start_time
        print(f"\nTotal Execution Time: {format_time_duration(execution_time)}")
        
        return {
            'allocation': allocation,
            'numerical_results': numerical_results,
            'execution_time': execution_time,
            'config': self.config
        }


def main():
    """Main function to run the greedy algorithm"""
    # Use same configuration as matching theory for fair comparison
    config = SystemConfiguration(
        num_users=10,
        num_servers=5,
        num_task_types=10,
        network_area_size=500.0,
        fixed_task_count=1000,
        random_seed=42,  # Fixed seed for reproducible comparison
        
        use_multilevel=True,
        local_processing_threshold=2.0
    )
    
    set_random_seeds(42)
    
    # Create and run the greedy algorithm
    greedy_algorithm = GreedyTaskOffloadingAlgorithm(config)
    
    print(f"\n=== Greedy Algorithm Configuration ===")
    print(f"Strategy: Distance-based (always choose nearest server)")
    print(f"No preference optimization")
    print(f"No dynamic load balancing")
    print(f"={'='*50}\n")
    
    results = greedy_algorithm.run_complete_algorithm()
    
    # Show distribution summary
    if hasattr(config, 'use_multilevel') and config.use_multilevel:
        print(f"\n=== Greedy Algorithm Results Summary ===")
        allocation = results['allocation']
        
        edge_tasks = sum(len(tasks) for server_id, tasks in allocation.items() if server_id.startswith('E'))
        regional_tasks = sum(len(tasks) for server_id, tasks in allocation.items() if server_id.startswith('R'))
        cloud_tasks = sum(len(tasks) for server_id, tasks in allocation.items() if server_id.startswith('C'))
        
        total_tasks = edge_tasks + regional_tasks + cloud_tasks
        
        print(f"Task distribution (greedy distance-based):")
        print(f"  🏢 Edge fog servers: {edge_tasks}/{total_tasks} tasks ({(edge_tasks/total_tasks)*100:.1f}%)")
        print(f"  🏗️  Regional fog servers: {regional_tasks}/{total_tasks} tasks ({(regional_tasks/total_tasks)*100:.1f}%)")
        print(f"  ☁️  Cloud servers: {cloud_tasks}/{total_tasks} tasks ({(cloud_tasks/total_tasks)*100:.1f}%)")
        print(f"\n⚠️  Note: Greedy algorithm tends to overload nearby servers!")
        print(f"={'='*50}")


if __name__ == "__main__":
    main()
