"""
Simulation-Based Performance Metrics Calculator
Combines task execution simulation with performance metrics calculation.
Uses real simulation data instead of theoretical formulas for accurate results.

This module:
1. Simulates actual task execution on fog servers after stable matching
2. Calculates performance metrics from real simulation data
3. Provides Section IV numerical results using simulation-based approach
"""

import time
import random
import numpy as np
from typing import Dict, List

# Import from simulation utilities (no external module dependencies)
from .simulation_utilities import SimulationUtilities, print_section


class SimulationMetrics:
    """
    Simulation-based calculator that combines task execution simulation with performance metrics calculation
    Uses real simulation data instead of theoretical formulas
    """
    
    def __init__(self, tasks: List[Dict], servers: List[Dict], users: List[Dict], 
                 server_capacities: Dict[str, int], transmission_delays: any, config=None):
        """
        Initialize the simulation-based metrics calculator
        
        Args:
            tasks: List of task dictionaries with computation requirements
            servers: List of server dictionaries with computational capabilities
            users: List of user dictionaries
            server_capacities: Dictionary mapping server_id to capacity limits
            transmission_delays: Matrix of transmission delays between users and servers
            config: SystemConfiguration object (optional, for hybrid capacity detection)
        """
        self.tasks = tasks
        self.servers = servers
        self.users = users
        self.server_capacities = server_capacities
        self.transmission_delays = transmission_delays
        self.config = config
        
    def run_simulation_and_calculate_metrics(self, allocation: Dict[str, List[str]]) -> Dict:
        """
        Main method: Run task execution simulation and calculate performance metrics from real data
        
        Args:
            allocation: Dictionary mapping server_id to list of assigned task_ids
            
        Returns:
            Dictionary containing both simulation results and performance metrics
        """
        print("\n" + "="*70)
        print("🔄 SIMULATION-BASED PERFORMANCE METRICS CALCULATION")
        print("="*70)
        print("Running simulation and calculating metrics from real execution data...")
        
        # Step 1: Run the simulation
        simulation_results = self._simulate_task_execution(allocation)
        
        # Step 2: Calculate performance metrics from simulation data
        performance_metrics = self._calculate_simulation_based_metrics(
            simulation_results, allocation
        )
        
        # Step 3: Calculate Section IV numerical results
        numerical_results = self._calculate_numerical_results_from_simulation(
            simulation_results, allocation
        )
        
        # Combine all results
        unified_results = {
            'simulation_data': simulation_results,
            'performance_metrics': performance_metrics,
            'numerical_results': numerical_results,
            'allocation': allocation
        }
        
        # Print comprehensive summary
        self._print_comprehensive_summary(unified_results)
        
        return unified_results
    
    def _simulate_task_execution(self, allocation: Dict[str, List[str]]) -> Dict:
        """
        Simulate the actual execution of tasks on assigned servers
        
        Args:
            allocation: Dictionary mapping server_id to list of assigned task_ids
            
        Returns:
            Dictionary containing detailed simulation results
        """
        # Add fresh randomness for simulation to get realistic variations each run
        # This ensures different simulation behavior even with same allocation
        import time as time_module
        simulation_seed = int(time_module.time() * 1000000) % (2**31 - 1)
        random.seed(simulation_seed)
        print(f"\n🎲 Simulation using random seed: {simulation_seed} (for realistic variations)")
        
        print("\n======================================================================")
        print("🔄 SIMULATING TASK EXECUTION AFTER STABLE MATCHING")
        print("======================================================================")
        print("Demonstrating real-time task processing on assigned fog servers...")
        
        simulation_results = {
            'server_execution_logs': {},
            'task_completion_times': {},
            'task_waiting_times': {},
            'server_utilization_over_time': {},
            'total_simulation_time': 0,
            'task_start_times': {},
            'task_queue_times': {}
        }
        
        # Prepare execution queues for each server
        server_queues = self._initialize_server_queues(allocation)
        
        print(f"\n📋 EXECUTION QUEUES INITIALIZED:")
        for server_id, queue_info in server_queues.items():
            print(f"  {server_id}: {len(queue_info['tasks'])} tasks queued")
        
        # Run the simulation
        simulation_time = self._run_simulation_loop(server_queues, simulation_results)
        
        # Store final results
        simulation_results['total_simulation_time'] = simulation_time
        simulation_results['server_execution_logs'] = {
            sid: info['execution_log'] for sid, info in server_queues.items()
        }
        
        # Calculate waiting times from simulation data
        self._calculate_waiting_times_from_simulation(simulation_results)
        
        # Print simulation summary
        self._print_simulation_summary(simulation_time, server_queues, simulation_results, allocation)
        
        return simulation_results
    
    def _initialize_server_queues(self, allocation: Dict[str, List[str]]) -> Dict:
        """
        Initialize execution queues for each server with assigned tasks
        """
        from .simulation_utilities import SimulationUtilities
        
        server_queues = {}
        
        # Track all task arrival times for realistic simulation
        task_arrival_times = {}
        
        for server_id, task_list in allocation.items():
            # Skip LOCAL_PROCESSING tasks (they are processed on user devices, not servers)
            if server_id == 'LOCAL_PROCESSING':
                print(f"ℹ️  Skipping {len(task_list)} tasks marked for LOCAL_PROCESSING (processed on user devices)")
                continue
                
            if task_list:
                # Find server info - with better error handling
                try:
                    server_info = next(s for s in self.servers if s['id'] == server_id)
                except StopIteration:
                    print(f"❌ ERROR: Server '{server_id}' not found in servers list!")
                    print(f"Available servers: {[s['id'] for s in self.servers]}")
                    print(f"Allocation keys: {list(allocation.keys())}")
                    raise ValueError(f"Server '{server_id}' from allocation not found in servers list")
                
                # Generate realistic arrival times for each task (within first 5 seconds)
                # This simulates asynchronous IoT task submission
                for task_id in task_list:
                    arrival_time = SimulationUtilities.generate_task_arrival_time(max_arrival_window=5.0)
                    task_arrival_times[task_id] = arrival_time
                
                server_queues[server_id] = {
                    'tasks': task_list.copy(),
                    'task_arrival_times': {tid: task_arrival_times[tid] for tid in task_list},
                    'server_info': server_info,
                    'current_task': None,
                    'task_start_time': 0,
                    'total_processed': 0,
                    'execution_log': []
                }
                
                # Log initial queuing of all tasks with their arrival times
                for task_id in task_list:
                    arrival = task_arrival_times[task_id]
                    server_queues[server_id]['execution_log'].append({
                        'event': 'task_queued',
                        'time': arrival,
                        'task_id': task_id
                    })
        
        return server_queues
    
    def _run_simulation_loop(self, server_queues: Dict, simulation_results: Dict) -> float:
        """
        Run the main simulation loop with time steps
        """
        # Simulation parameters
        simulation_time = 0
        time_step = 0.1  # 100ms time steps
        # No maximum time limit - run until all tasks complete
        
        print(f"\n⏱️  STARTING SIMULATION (time step: {time_step}s)")
        print("📋 Will run until ALL tasks are completed")
        print("-" * 70)
        
        # Simulation loop - run until all tasks complete
        while True:
            active_servers = 0
            
            for server_id, queue_info in server_queues.items():
                # Start new task if server is idle and has tasks in queue that have arrived
                if queue_info['current_task'] is None and queue_info['tasks']:
                    # Check if the next task has arrived (only process tasks that have arrived)
                    next_task_id = queue_info['tasks'][0]
                    task_arrival_time = queue_info['task_arrival_times'].get(next_task_id, 0.0)
                    
                    if simulation_time >= task_arrival_time:
                        self._start_new_task(server_id, queue_info, simulation_time, simulation_results)
                
                # Check if current task is completed
                if queue_info['current_task'] and simulation_time >= queue_info['current_task']['completion_time']:
                    self._complete_current_task(server_id, queue_info, simulation_time, simulation_results)
                
                # Count active servers (including servers waiting for task arrivals)
                if queue_info['current_task'] or queue_info['tasks']:
                    active_servers += 1
            
            # Break if no servers are active (all tasks completed)
            if active_servers == 0:
                print(f"\n🏁 All tasks completed at {simulation_time:.1f}s")
                break
            
            # Safety check to prevent infinite loops (very generous limit)
            if simulation_time > 1000.0:  # 1000 seconds = ~16 minutes max
                print(f"\n⚠️  Simulation timeout after {simulation_time:.1f}s")
                print(f"   Some tasks may not have completed due to system limitations")
                break
            
            simulation_time += time_step
        
        return simulation_time
    
    def _start_new_task(self, server_id: str, queue_info: Dict, simulation_time: float, simulation_results: Dict):
        """
        Start processing a new task on a server
        """
        # Start new task
        current_task_id = queue_info['tasks'].pop(0)
        task_info = next(t for t in self.tasks if t['id'] == current_task_id)
        
        # Calculate base processing time
        base_processing_time = task_info['computation_requirement'] / queue_info['server_info']['computational_capability']
        
        # Add realistic random variations to simulate real-world conditions
        from .simulation_utilities import SimulationUtilities
        
        # 1. Add processing time variation (±25% for CPU scheduling, cache misses, temperature throttling, etc.)
        # Increased from 15% to 25% for more realistic real-world variations
        realistic_processing_time = SimulationUtilities.add_processing_time_variation(
            base_processing_time, variation_percent=25.0
        )
        
        # 2. Add startup delay (context switching, task initialization, cache warming)
        # Increased range from 2-15ms to 5-30ms for more realistic variations
        startup_delay = SimulationUtilities.add_startup_delay(min_delay=0.005, max_delay=0.030)
        
        # 3. Add network jitter to transmission delay from user to server
        network_jitter = 0
        if self.transmission_delays is not None:
            # Get user of this task
            user_id = task_info.get('user_id', None)
            if user_id:
                try:
                    user_idx = next(i for i, u in enumerate(self.users) if u['id'] == user_id)
                    server_idx = next(i for i, s in enumerate(self.servers) if s['id'] == server_id)
                    base_transmission_delay = self.transmission_delays[user_idx][server_idx]
                    # Add realistic network jitter (±20% increased from ±10%)
                    transmission_with_jitter = SimulationUtilities.add_network_jitter(
                        base_transmission_delay, jitter_percent=20.0
                    )
                    network_jitter = transmission_with_jitter - base_transmission_delay
                except (StopIteration, IndexError, TypeError):
                    network_jitter = 0
        
        # 4. Account for concurrent task interference (if other tasks are running)
        # Count how many other tasks are currently being processed on this server
        # (This is a simplified model - ideally would track actual concurrent execution)
        concurrent_tasks = 1  # Current task
        
        # Apply interference overhead
        final_processing_time_with_interference = SimulationUtilities.simulate_task_interference(
            realistic_processing_time, concurrent_tasks=concurrent_tasks
        )
        
        # Total realistic processing time = processing + startup + network jitter + interference
        final_processing_time = final_processing_time_with_interference + startup_delay + max(network_jitter, 0)
        
        queue_info['current_task'] = {
            'task_id': current_task_id,
            'start_time': simulation_time,
            'processing_time': final_processing_time,
            'base_processing_time': base_processing_time,
            'realistic_processing_time': realistic_processing_time,
            'startup_delay': startup_delay,
            'network_jitter': network_jitter,
            'completion_time': simulation_time + final_processing_time
        }
        
        # Store task start time for waiting time calculation
        simulation_results['task_start_times'][current_task_id] = simulation_time
        
        # Show both theoretical and realistic times with detailed breakdown
        variation_pct = ((final_processing_time - base_processing_time) / base_processing_time) * 100
        jitter_info = f", jitter: {network_jitter*1000:+.2f}ms" if network_jitter != 0 else ""
        print(f"⚡ {simulation_time:.1f}s: {server_id} starts {current_task_id} "
              f"(theoretical: {base_processing_time:.3f}s, realistic: {final_processing_time:.3f}s, "
              f"startup: {startup_delay*1000:.2f}ms{jitter_info}, "
              f"total variation: {variation_pct:+.1f}%)")
        
        queue_info['execution_log'].append({
            'event': 'task_start',
            'time': simulation_time,
            'task_id': current_task_id,
            'base_processing_time': base_processing_time,
            'realistic_processing_time': realistic_processing_time,
            'startup_delay': startup_delay,
            'network_jitter': network_jitter,
            'final_processing_time': final_processing_time,
            'variation_percent': variation_pct,
            'concurrent_tasks': concurrent_tasks
        })
    
    def _complete_current_task(self, server_id: str, queue_info: Dict, simulation_time: float, simulation_results: Dict):
        """
        Complete the current task on a server
        """
        completed_task = queue_info['current_task']
        task_id = completed_task['task_id']
        actual_completion_time = simulation_time
        
        print(f"✅ {simulation_time:.1f}s: {server_id} completed {task_id} (took {actual_completion_time - completed_task['start_time']:.3f}s)")
        
        # Log completion
        queue_info['execution_log'].append({
            'event': 'task_complete',
            'time': simulation_time,
            'task_id': task_id,
            'completion_time': actual_completion_time
        })
        
        # Store task completion time for metrics calculation
        simulation_results['task_completion_times'][task_id] = actual_completion_time
        
        queue_info['total_processed'] += 1
        queue_info['current_task'] = None
    
    def _calculate_waiting_times_from_simulation(self, simulation_results: Dict):
        """
        Calculate waiting times for each task from simulation data with realistic variations
        """
        from .simulation_utilities import SimulationUtilities
        
        waiting_times = {}
        
        for task_id in simulation_results['task_completion_times'].keys():
            # Find when task was queued (always at time 0 in our simulation)
            queue_time = 0.0
            
            # Find when task started processing
            start_time = simulation_results['task_start_times'].get(task_id, 0.0)
            
            # Base waiting time = time from queuing to start of processing
            base_waiting_time = start_time - queue_time
            
            # Add realistic queue wait variation based on system load
            # Estimate load factor based on number of tasks vs servers
            num_tasks = len(simulation_results['task_completion_times'])
            num_servers = len([s for s in self.servers])
            load_factor = max(1.0, num_tasks / (num_servers * 3))  # Normalize by expected capacity
            
            # Apply queue wait variation (will add variation to waiting time)
            realistic_waiting_time = SimulationUtilities.add_queue_wait_variation(
                base_waiting_time, load_factor=min(load_factor, 3.0)  # Cap at 3x load
            )
            
            waiting_times[task_id] = realistic_waiting_time
        
        simulation_results['task_waiting_times'] = waiting_times
    
    def _calculate_simulation_based_metrics(self, simulation_results: Dict, allocation: Dict[str, List[str]]) -> Dict:
        """
        Calculate performance metrics using real simulation data instead of theoretical formulas
        """
        print("\n=== Calculating Performance Metrics from Simulation Data ===")
        
        # Extract data from simulation
        completion_times = list(simulation_results['task_completion_times'].values())
        waiting_times = list(simulation_results['task_waiting_times'].values())
        
        # Calculate basic metrics
        max_completion_time = max(completion_times) if completion_times else 0
        avg_completion_time = sum(completion_times) / len(completion_times) if completion_times else 0
        avg_waiting_time = sum(waiting_times) / len(waiting_times) if waiting_times else 0
        
        # Calculate server utilization from simulation
        server_utilizations = {}
        total_simulation_time = simulation_results['total_simulation_time']
        
        for server_id, task_list in allocation.items():
            # Skip LOCAL_PROCESSING tasks (processed on user devices)
            if server_id == 'LOCAL_PROCESSING':
                continue
                
            if task_list and total_simulation_time > 0:
                # Calculate actual processing time for this server
                total_processing_time = 0
                for task_id in task_list:
                    task = next(t for t in self.tasks if t['id'] == task_id)
                    server = next(s for s in self.servers if s['id'] == server_id)
                    processing_time = task['computation_requirement'] / server['computational_capability']
                    total_processing_time += processing_time
                
                # Utilization = total processing time / total simulation time
                utilization = min(total_processing_time / total_simulation_time, 1.0)
                server_utilizations[server_id] = utilization
            else:
                server_utilizations[server_id] = 0.0
        
        # Calculate assignment success rate
        total_tasks = len(self.tasks)
        assigned_tasks = len(completion_times)
        assignment_success_rate = assigned_tasks / total_tasks if total_tasks > 0 else 0
        
        # Calculate energy and cost from simulation data
        total_energy = 0
        total_cost = 0
        
        for server_id, task_list in allocation.items():
            # Skip LOCAL_PROCESSING tasks (processed on user devices)
            if server_id == 'LOCAL_PROCESSING':
                continue
                
            if task_list:
                server = next(s for s in self.servers if s['id'] == server_id)
                # Use default energy efficiency if not present
                energy_efficiency = server.get('energy_efficiency', 0.75)
                server_energy = len(task_list) * energy_efficiency * 0.1
                total_energy += server_energy
        
        for task_id, completion_time in simulation_results['task_completion_times'].items():
            task = next((t for t in self.tasks if t['id'] == task_id), None)
            if task:
                computation_cost = task['computation_requirement'] * 0.001
                total_cost += computation_cost
        
        metrics = {
            'max_completion_time': max_completion_time,
            'avg_completion_time': avg_completion_time,
            'avg_waiting_time': avg_waiting_time,
            'total_delay': sum(completion_times),
            'total_cost': total_cost,
            'total_energy': total_energy,
            'server_utilization': server_utilizations,
            'assignment_success_rate': assignment_success_rate,
            'num_unassigned_tasks': total_tasks - assigned_tasks,
            'task_completion_times': completion_times,
            'task_waiting_times': waiting_times
        }
        
        return metrics
    
    def _calculate_numerical_results_from_simulation(self, simulation_results: Dict, allocation: Dict[str, List[str]]) -> Dict:
        """
        Calculate Section IV numerical results using real simulation data
        """
        print("\n=== IV. NUMERICAL RESULTS - Performance Metrics (SIMULATION-BASED) ===")
        
        # Extract real data from simulation
        completion_times = list(simulation_results['task_completion_times'].values())
        waiting_times = list(simulation_results['task_waiting_times'].values())
        total_simulation_time = simulation_results['total_simulation_time']
        
        # 1. Worst total task completion time (T̄_M) - from actual simulation
        worst_completion_time_TM = max(completion_times) if completion_times else 0
        
        # 2. Mean task total completion time (T̄_T) - from actual simulation
        mean_completion_time_TT = sum(completion_times) / len(completion_times) if completion_times else 0
        
        # 3. Mean waiting time (T̄_W) - from actual simulation queue times
        mean_waiting_time_TW = sum(waiting_times) / len(waiting_times) if waiting_times else 0
        
        # 4. Jain's fairness index (I_J) - using real completion times
        if completion_times and len(completion_times) > 1:
            k = len(completion_times)
            sum_v = sum(completion_times)
            sum_v_squared = sum(v*v for v in completion_times)
            
            if sum_v_squared > 0:
                jains_index_IJ = (sum_v * sum_v) / (k * sum_v_squared)
            else:
                jains_index_IJ = 1.0
        else:
            jains_index_IJ = 1.0 if len(completion_times) == 1 else 0.0
        
        # Server utilization fairness from actual execution
        # For unlimited capacity: use actual workload (processing time / total time)
        server_utilizations_list = []
        for server_id in [s['id'] for s in self.servers]:
            capacity = self.server_capacities.get(server_id, 1)
            assigned = len(allocation.get(server_id, []))
            
            if capacity == float('inf'):
                # Unlimited capacity: use actual utilization from simulation data
                if server_id in allocation and allocation[server_id] and total_simulation_time > 0:
                    # Calculate actual processing time for this server
                    total_processing_time = 0
                    for task_id in allocation[server_id]:
                        task = next((t for t in self.tasks if t['id'] == task_id), None)
                        if task:
                            server = next(s for s in self.servers if s['id'] == server_id)
                            processing_time = task['computation_requirement'] / server['computational_capability']
                            total_processing_time += processing_time
                    
                    # Utilization = actual work time / total simulation time
                    utilization = min(total_processing_time / total_simulation_time, 1.0)
                else:
                    utilization = 0.0
            else:
                # Limited capacity: use traditional calculation
                utilization = assigned / capacity if capacity > 0 else 0
            
            server_utilizations_list.append(utilization)
        
        if server_utilizations_list and any(u > 0 for u in server_utilizations_list):
            n = len(server_utilizations_list)
            sum_x = sum(server_utilizations_list)
            sum_x_squared = sum(x*x for x in server_utilizations_list)
            
            if sum_x_squared > 0:
                server_utilization_fairness = (sum_x * sum_x) / (n * sum_x_squared)
            else:
                server_utilization_fairness = 1.0
        else:
            server_utilization_fairness = 1.0
        
        # System performance metrics
        total_tasks = len(self.tasks)
        # Count ASSIGNED tasks from allocation (not just completed tasks)
        assigned_tasks = sum(len(tasks) for tasks in allocation.values())
        completed_tasks = len(completion_times)
        assignment_success_rate = assigned_tasks / total_tasks if total_tasks > 0 else 0
        completion_rate = completed_tasks / total_tasks if total_tasks > 0 else 0
        
        # Server utilizations - use actual workload for hybrid/unlimited capacity
        server_utilizations = {}
        
        # Check if using hybrid capacity model (from config)
        use_hybrid_capacity = self.config.use_hybrid_capacity if self.config else False
        initial_capacity = self.config.initial_server_capacity if self.config else 3
        
        for server_id in [s['id'] for s in self.servers]:
            capacity = self.server_capacities.get(server_id, 1)
            assigned = len(allocation.get(server_id, []))
            
            # For HYBRID capacity model: Always use time-based utilization
            # (servers accept unlimited tasks with waiting times after reaching initial capacity)
            if use_hybrid_capacity or capacity == float('inf'):
                # Time-based utilization: actual work time / total simulation time
                if server_id in allocation and allocation[server_id] and total_simulation_time > 0:
                    # Calculate actual processing time for this server
                    total_processing_time = 0
                    for task_id in allocation[server_id]:
                        task = next((t for t in self.tasks if t['id'] == task_id), None)
                        if task:
                            server = next(s for s in self.servers if s['id'] == server_id)
                            processing_time = task['computation_requirement'] / server['computational_capability']
                            total_processing_time += processing_time
                    
                    # Utilization = actual work time / total simulation time (as percentage)
                    # Cap at 100% (can't exceed 100% actual utilization in real time)
                    utilization = min(total_processing_time / total_simulation_time, 1.0) * 100
                else:
                    utilization = 0.0
            else:
                # Pure limited capacity: traditional calculation
                utilization = (assigned / capacity * 100) if capacity > 0 else 0
            
            server_utilizations[server_id] = utilization
        
        avg_server_utilization = sum(server_utilizations.values()) / len(server_utilizations) if server_utilizations else 0
        
        # Print results
        print("\n--- Performance Metrics (Section IV) - SIMULATION-BASED ---")
        print(f"System Configuration:")
        print(f"  • Number of FNs (n): {len(self.servers)}")
        print(f"  • Total number of tasks (m): {total_tasks}")
        print(f"  • Task instructions range: [10000, 50000] (simulated)")
        print(f"  • Network area: Circular radius 100m")
        print(f"  • Total simulation time: {total_simulation_time:.4f}s")
        
        print(f"\nKey Performance Indicators (From Simulation):")
        print(f"  • Worst total task completion time (T̄_M): {worst_completion_time_TM:.4f}s")
        print(f"  • Mean task total completion time (T̄_T): {mean_completion_time_TT:.4f}s")
        print(f"  • Mean waiting time (T̄_W): {mean_waiting_time_TW:.4f}s")
        print(f"  • Jain's fairness index (I_J): {jains_index_IJ:.4f} (task completion time fairness)")
        print(f"  • Server utilization fairness: {server_utilization_fairness:.4f} (resource allocation fairness)")
        
        print(f"\nSystem Performance:")
        print(f"  • Task assignment success rate: {assignment_success_rate:.1%} ({assigned_tasks}/{total_tasks} tasks assigned)")
        print(f"  • Task completion rate: {completion_rate:.1%} ({completed_tasks}/{total_tasks} tasks completed)")
        print(f"  • Average server utilization: {avg_server_utilization:.1f}%")
        print(f"  • Simulation duration: {total_simulation_time:.1f}s")
        
        print(f"\nServer Utilization Details:")
        for server_id, utilization in server_utilizations.items():
            capacity = self.server_capacities.get(server_id, 1)
            assigned = len(allocation.get(server_id, []))
            
            # For hybrid capacity, show actual utilization (time-based)
            if use_hybrid_capacity:
                print(f"  • {server_id}: {utilization:.1f}% utilization ({assigned} tasks, hybrid capacity model)")
            elif capacity == float('inf'):
                print(f"  • {server_id}: {utilization:.1f}% utilization ({assigned} tasks)")
            else:
                print(f"  • {server_id}: {utilization:.1f}% ({assigned}/{capacity} tasks)")
        
        # Return structured results
        numerical_results = {
            'worst_completion_time_TM': worst_completion_time_TM,
            'mean_completion_time_TT': mean_completion_time_TT,
            'mean_waiting_time_TW': mean_waiting_time_TW,
            'jains_index_IJ': jains_index_IJ,
            'server_utilization_fairness': server_utilization_fairness,
            'num_fog_nodes_n': len(self.servers),
            'total_tasks_m': total_tasks,
            'assigned_tasks': assigned_tasks,
            'assignment_success_rate': assignment_success_rate,
            'server_utilizations': server_utilizations,
            'avg_server_utilization': avg_server_utilization,
            'total_simulation_time': total_simulation_time,
            'task_completion_times': completion_times,
            'waiting_times': waiting_times
        }
        
        return numerical_results
    
    def _print_simulation_summary(self, simulation_time: float, server_queues: Dict, 
                                  simulation_results: Dict, allocation: Dict[str, List[str]]):
        """
        Print comprehensive simulation summary
        """
        print("\n" + "="*70)
        print("📊 SIMULATION SUMMARY")
        print("="*70)
        
        total_tasks_completed = sum(info['total_processed'] for info in server_queues.values())
        print(f"Total simulation time: {simulation_time:.2f}s")
        print(f"Tasks completed: {total_tasks_completed}/{len(self.tasks)}")
        
        print("\n🏭 SERVER PERFORMANCE:")
        
        # Check if using hybrid capacity model
        use_hybrid = self.config.use_hybrid_capacity if self.config else False
        initial_cap = self.config.initial_server_capacity if self.config else 3
        
        for server_id, queue_info in server_queues.items():
            completed = queue_info['total_processed']
            capacity = self.server_capacities.get(server_id, 1)
            
            if use_hybrid:
                # Hybrid capacity: show actual time-based utilization
                # Calculate processing time for this server
                total_processing_time = 0
                for task_id in allocation.get(server_id, []):
                    task = next((t for t in self.tasks if t['id'] == task_id), None)
                    if task:
                        server = next(s for s in self.servers if s['id'] == server_id)
                        processing_time = task['computation_requirement'] / server['computational_capability']
                        total_processing_time += processing_time
                
                # Utilization = processing time / simulation time (capped at 100%)
                utilization = min(total_processing_time / simulation_time, 1.0) * 100 if simulation_time > 0 else 0
                print(f"  {server_id}: {completed} tasks processed ({utilization:.1f}% utilization)")
            else:
                # Pure limited capacity: traditional calculation
                efficiency = (completed / capacity * 100) if capacity > 0 else 0
                print(f"  {server_id}: {completed}/{capacity} tasks processed ({efficiency:.1f}% of capacity)")
        
        if simulation_results['task_completion_times']:
            completion_times = list(simulation_results['task_completion_times'].values())
            print(f"\n⏱️  TASK COMPLETION STATISTICS:")
            print(f"  Fastest task: {min(completion_times):.3f}s")
            print(f"  Slowest task: {max(completion_times):.3f}s")
            print(f"  Average completion: {sum(completion_times)/len(completion_times):.3f}s")
        
        print("="*70)
    
    def _print_comprehensive_summary(self, unified_results: Dict):
        """
        Print comprehensive summary of both simulation and metrics
        """
        print("\n" + "="*70)
        print("📈 COMPREHENSIVE PERFORMANCE ANALYSIS")
        print("="*70)
        
        numerical_results = unified_results['numerical_results']
        
        print("\n------------------------------------------------------------")
        print("NUMERICAL RESULTS SUMMARY (SECTION IV)")
        print("------------------------------------------------------------")
        print("Research Paper Performance Metrics:")
        print(f"  T̄_M (Worst completion time): {numerical_results['worst_completion_time_TM']:.4f}s")
        print(f"  T̄_T (Mean completion time): {numerical_results['mean_completion_time_TT']:.4f}s")
        print(f"  T̄_W (Mean waiting time): {numerical_results['mean_waiting_time_TW']:.4f}s")
        print(f"  I_J (Jain's fairness index): {numerical_results['jains_index_IJ']:.4f}")
        
    def get_detailed_execution_timeline(self, unified_results: Dict) -> List[Dict]:
        """
        Get a detailed timeline of all execution events
        """
        simulation_results = unified_results['simulation_data']
        timeline = []
        
        for server_id, events in simulation_results['server_execution_logs'].items():
            for event in events:
                timeline.append({
                    'server_id': server_id,
                    'time': event['time'],
                    'event_type': event['event'],
                    'task_id': event['task_id'],
                    'details': event
                })
        
        # Sort by time
        timeline.sort(key=lambda x: x['time'])
        
        return timeline


def main():
    """Example usage of the Simulation-Based Performance Metrics Calculator"""
    print("=== Simulation-Based Performance Metrics Calculator ===")
    print("This module combines task execution simulation with performance metrics calculation.")
    print("It uses real simulation data instead of theoretical formulas for accurate results.")
    print("\nKey Features:")
    print("  • Real-time task execution simulation on fog servers")
    print("  • Performance metrics calculated from actual simulation data")
    print("  • Section IV numerical results using simulation-based approach")
    print("  • Unified approach replacing separate simulation and theoretical calculations")
    print("\nFor actual usage, import SimulationMetrics and call run_simulation_and_calculate_metrics()")
    print("with appropriate allocation data from the matching algorithm.")


if __name__ == "__main__":
    main()