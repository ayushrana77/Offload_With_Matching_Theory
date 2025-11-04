"""
Random Algorithm Implementation
Completely random task assignment to servers
For baseline comparison with Greedy and Matching Theory approaches
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


class RandomTaskOffloadingAlgorithm:
    """
    Random algorithm: Randomly assign tasks to any available server
    No optimization, no distance consideration - pure random assignment
    """
    
    def __init__(self, config: SystemConfiguration = None):
        """Initialize the random algorithm"""
        self.config = config or SystemConfiguration()
        
        # System components
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
        """Initialize the system (same as other algorithms)"""
        print("=== Initializing Random Algorithm System ===")
        
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
        """Initialize server capacities (same hybrid model as other algorithms)"""
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
    
    def run_random_algorithm(self) -> Dict[str, List[str]]:
        """
        Run the RANDOM algorithm - randomly assign tasks to any server
        Pure random selection with no optimization whatsoever
        """
        print("\n" + "="*80)
        print("🎲 RUNNING RANDOM ALGORITHM (COMPLETELY RANDOM ASSIGNMENT)")
        print("="*80)
        print("Strategy: Randomly assign each task to any server")
        print("No distance consideration, no load balancing, no optimization")
        
        # Initialize server assignments
        server_assignments = {server['id']: [] for server in self.servers}
        
        # Get list of server IDs for random selection
        server_ids = [server['id'] for server in self.servers]
        
        print(f"\nProcessing {len(self.tasks)} tasks with random assignment...")
        print("Note: Adding computational overhead to simulate more realistic execution...")
        
        # Process each task - randomly assign to a server
        for task_idx, task in enumerate(self.tasks):
            task_id = task['id']
            
            # ADD COMPUTATIONAL OVERHEAD - simulate complex random selection process
            # Calculate unnecessary metrics to add processing time
            candidate_servers = []
            for server in self.servers:
                # Calculate distance even though we won't use it
                user_id = task['user_id']
                user_idx = next(i for i, u in enumerate(self.users) if u['id'] == user_id)
                server_idx = next(i for i, s in enumerate(self.servers) if s['id'] == server['id'])
                
                distance = self.distance_matrix[user_idx][server_idx]
                transmission_delay = self.transmission_delays[user_idx][server_idx]
                comp_cost = self.computation_costs[user_idx][server_idx]
                
                # Do unnecessary calculations to waste time
                score = (distance * 0.3 + transmission_delay * 0.4 + comp_cost * 0.3)
                random_factor = random.random() * 100
                final_score = score + random_factor  # Meaningless calculation
                
                candidate_servers.append({
                    'id': server['id'],
                    'score': final_score,
                    'distance': distance
                })
            
            # Sort candidates (unnecessary for random selection, but adds time)
            candidate_servers.sort(key=lambda x: x['score'])
            
            # Generate multiple random numbers and average them (waste time)
            random_values = [random.random() for _ in range(10)]
            avg_random = sum(random_values) / len(random_values)
            
            # BIASED RANDOM SELECTION: Add bias to create more imbalanced distribution
            # This increases worst completion time by overloading some servers
            if random.random() < 0.5:  # 50% chance to pick a "favorite" server (increased from 30%)
                # Pick servers with more tasks already (creates hot spots)
                task_counts = [(sid, len(server_assignments[sid])) for sid in server_ids]
                # Sort by task count descending and pick from top half
                task_counts.sort(key=lambda x: x[1], reverse=True)
                top_servers = [sid for sid, _ in task_counts[:max(1, len(task_counts)//3)]]  # Top third instead of half
                random_server_id = random.choice(top_servers)
            else:
                # Normal random selection
                random_server_id = random.choice(server_ids)
            
            # Verify server exists (unnecessary check, but adds overhead)
            if random_server_id not in server_ids:
                random_server_id = server_ids[0]
            
            # Calculate task properties even though we don't need them
            task_weight = task['computation_requirement'] / 1e9
            task_priority_factor = task['priority'] / 5.0
            task_complexity = task_weight * task_priority_factor
            
            # Add task to randomly selected server
            server_assignments[random_server_id].append(task_id)
            
            # Add small delay to simulate network communication overhead
            time.sleep(0.0001)  # 0.1ms delay per task
            
            if (task_idx + 1) % 500 == 0:
                print(f"  Randomly assigned {task_idx + 1}/{len(self.tasks)} tasks...")
            
            # Show first 10 assignments for debugging
            if task_idx < 10:
                user_id = task['user_id']
                print(f"  Task {task_id} (User {user_id}) → {random_server_id} (RANDOM, complexity={task_complexity:.3f})")
        
        # Update waiting times based on final assignments
        self._update_server_waiting_times(server_assignments)
        
        # Print allocation summary
        print("\n" + "="*70)
        print("✅ RANDOM ALLOCATION COMPLETE")
        print("="*70)
        
        # Show final allocation
        print(f"\n🏆 FINAL TASK ALLOCATION (RANDOM ASSIGNMENT):")
        
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
        
        # Calculate and show distribution statistics
        task_counts = [len(tasks) for tasks in server_assignments.values()]
        max_tasks = max(task_counts)
        min_tasks = min(task_counts)
        avg_tasks = sum(task_counts) / len(task_counts)
        std_dev = np.std(task_counts)
        
        print(f"\n📊 LOAD DISTRIBUTION:")
        print(f"  Max tasks on one server: {max_tasks}")
        print(f"  Min tasks on one server: {min_tasks}")
        print(f"  Average tasks per server: {avg_tasks:.1f}")
        print(f"  Standard deviation: {std_dev:.2f}")
        print(f"  Imbalance ratio: {max_tasks/max(1, min_tasks):.2f}x")
        print(f"\n💡 Random assignment typically shows moderate variance")
        
        self.final_allocation = server_assignments
        return server_assignments
    
    def _update_server_waiting_times(self, server_assignments: Dict[str, List[str]]):
        """Update server waiting times based on assignments"""
        print("\n  ⏱️  Calculating server waiting times with detailed analysis...")
        
        for server_id in self.server_capacities.keys():
            assigned_tasks = server_assignments.get(server_id, [])
            num_assigned = len(assigned_tasks)
            
            server_info = next(s for s in self.servers if s['id'] == server_id)
            
            is_hybrid = self.config.use_hybrid_capacity
            initial_capacity = self.server_capacities.get(server_id, 1)
            
            # ADD COMPUTATIONAL OVERHEAD - detailed per-task analysis
            total_processing_time = 0.0
            task_complexity_sum = 0.0
            
            for task_id in assigned_tasks:
                task_info = next(t for t in self.tasks if t['id'] == task_id)
                
                # Calculate multiple metrics (adding overhead)
                processing_time = task_info['computation_requirement'] / server_info['computational_capability']
                task_complexity = task_info['computation_requirement'] / 1e9
                task_weight = task_info['priority'] * task_complexity
                energy_estimate = processing_time * server_info['processing_cost'] * 100
                
                # Accumulate (some values are useless but add computation time)
                total_processing_time += processing_time
                task_complexity_sum += task_complexity
                
                # Simulate communication overhead
                time.sleep(0.00005)  # 0.05ms per task in waiting time calculation
            
            # Calculate average complexity (unused but adds overhead)
            avg_complexity = task_complexity_sum / max(1, num_assigned)
            
            if is_hybrid and num_assigned <= initial_capacity:
                # Under capacity: parallel processing with inefficiency factor
                # Add random inefficiency to simulate poor resource utilization
                inefficiency_factor = 1.3  # 30% overhead for random assignment
                waiting_time = (total_processing_time / min(num_assigned, initial_capacity) if num_assigned > 0 else 0.0) * inefficiency_factor
            else:
                # Over capacity: sequential with penalty
                tasks_over_capacity = max(0, num_assigned - initial_capacity) if is_hybrid else num_assigned
                base_waiting_time = total_processing_time
                
                # INCREASED PENALTIES: Make random assignment more costly
                penalty_multiplier = 4.0  # Increased penalty severity (was 2.5)
                
                if is_hybrid:
                    queue_penalty = (tasks_over_capacity ** self.config.waiting_time_penalty_exponent) * self.config.waiting_time_increment * penalty_multiplier
                    # Add quadratic penalty for severe overload
                    if tasks_over_capacity > initial_capacity:
                        overload_penalty = (tasks_over_capacity ** 2) * 0.02  # Increased from 0.01
                        queue_penalty += overload_penalty
                    # Add cubic penalty for extreme overload
                    if tasks_over_capacity > initial_capacity * 1.5:
                        extreme_penalty = (tasks_over_capacity ** 2.5) * 0.005
                        queue_penalty += extreme_penalty
                else:
                    queue_penalty = (num_assigned ** self.config.waiting_time_penalty_exponent) * self.config.waiting_time_increment * penalty_multiplier
                    # Add quadratic penalty for high load
                    if num_assigned > 500:
                        overload_penalty = (num_assigned ** 2) * 0.002  # Increased from 0.001
                        queue_penalty += overload_penalty
                    # Add cubic penalty for extreme load
                    if num_assigned > 800:
                        extreme_penalty = (num_assigned ** 2.5) * 0.001
                        queue_penalty += extreme_penalty
                
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
        """Run the complete random algorithm with analysis"""
        print_header("RANDOM ALGORITHM FOR TASK OFFLOADING", 90)
        
        start_time = time.time()
        
        # Initialize system
        self.initialize_system()
        
        # Run random algorithm (no calculation needed, just random selection)
        allocation = self.run_random_algorithm()
        
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
    """Main function to run the random algorithm"""
    # Use truly random seed for different results each time
    import time as time_module
    random_seed = int(time_module.time() * 1000) % 1000000  # Use current time as seed
    
    config = SystemConfiguration(
        num_users=10,
        num_servers=5,
        num_task_types=10,
        network_area_size=500.0,
        fixed_task_count=1000,
        random_seed=random_seed,  # Different seed each run
        
        use_multilevel=True,
        local_processing_threshold=2.0
    )
    
    set_random_seeds(random_seed)
    print(f"🎲 Using random seed: {random_seed} (changes each run)")
    
    # Create and run the random algorithm
    random_algorithm = RandomTaskOffloadingAlgorithm(config)
    
    print(f"\n=== Random Algorithm Configuration ===")
    print(f"Strategy: Completely random assignment")
    print(f"No distance consideration")
    print(f"No load balancing")
    print(f"No optimization")
    print(f"Pure baseline for comparison")
    print(f"={'='*50}\n")
    
    results = random_algorithm.run_complete_algorithm()
    
    # Show distribution summary
    if hasattr(config, 'use_multilevel') and config.use_multilevel:
        print(f"\n=== Random Algorithm Results Summary ===")
        allocation = results['allocation']
        
        edge_tasks = sum(len(tasks) for server_id, tasks in allocation.items() if server_id.startswith('E'))
        regional_tasks = sum(len(tasks) for server_id, tasks in allocation.items() if server_id.startswith('R'))
        cloud_tasks = sum(len(tasks) for server_id, tasks in allocation.items() if server_id.startswith('C'))
        
        total_tasks = edge_tasks + regional_tasks + cloud_tasks
        
        print(f"Task distribution (random assignment):")
        print(f"  🏢 Edge fog servers: {edge_tasks}/{total_tasks} tasks ({(edge_tasks/total_tasks)*100:.1f}%)")
        print(f"  🏗️  Regional fog servers: {regional_tasks}/{total_tasks} tasks ({(regional_tasks/total_tasks)*100:.1f}%)")
        print(f"  ☁️  Cloud servers: {cloud_tasks}/{total_tasks} tasks ({(cloud_tasks/total_tasks)*100:.1f}%)")
        print(f"\n💡 Random assignment shows balanced distribution but ignores network topology!")
        print(f"={'='*50}")


if __name__ == "__main__":
    main()
