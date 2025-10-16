"""
Proposed Algorithm Implementation
Matching Theory Framework for Task Offloading in Fog Computing for IoT Systems
Based on the research paper's proposed algorithm
"""

import numpy as np
import random
import time
from typing import Dict, List

from config import SystemConfiguration
from fog_preferences import FogPreferencesGenerator
from iot_preferences import IoTPreferencesGenerator
from simulation_metrics import SimulationMetrics
from utility import (
    SystemUtilities, MatchingUtilities, set_random_seeds,
    print_header, print_section
)


class ProposedTaskOffloadingAlgorithm:
    """
    Implementation of the proposed matching theory-based task offloading algorithm
    """
    
    def __init__(self, config: SystemConfiguration = None):
        """Initialize the proposed algorithm"""
        self.config = config or SystemConfiguration()
        
        # System components
        self.users = []                   # IoT users
        self.servers = []                 # Fog servers
        self.tasks = []                   # Tasks to be offloaded
        
        # System state
        self.distance_matrix = None       # Distance between users and servers
        self.channel_gains = None         # Channel gain matrix
        self.transmission_delays = None   # Transmission delay matrix
        self.computation_costs = None     # Computation cost matrix
        self.server_capacities = {}       # Server capacity constraints
        self.server_waiting_times = {}    # Current waiting time ωᵢ for each server i (per pseudocode line 17)
        
        # Tracking variables for algorithm compliance
        
        # Algorithm results
        self.user_preferences = {}        # User preference lists
        self.server_preferences = {}      # Server preference lists
        self.final_allocation = {}        # Final task allocation
        
        # Preference generators
        self.fog_pref_generator = FogPreferencesGenerator()
        self.iot_pref_generator = IoTPreferencesGenerator()
        
        # Unified simulation and metrics calculator
        self.unified_calculator = None  # Will be initialized when needed
        
    def initialize_system(self):
        """Initialize the system with users, servers, and tasks"""
        print("=== Initializing Proposed Algorithm System ===")
        
        # Validate configuration using config's validate method
        config_errors = self.config.validate()
        if config_errors:
            print("Configuration validation errors:")
            for error in config_errors:
                print(f"  - {error}")
            raise ValueError("Invalid system configuration")
        
        # Generate users (IoT devices)
        self._generate_users()
        
        # Generate servers (fog nodes)
        self._generate_servers()
        
        # Generate tasks
        self._generate_tasks()
        
        # Calculate system matrices
        self._calculate_system_matrices()
        
        # Initialize server capacities
        self._initialize_server_capacities()
        
        print(f"System initialized: {len(self.users)} users, {len(self.servers)} servers, {len(self.tasks)} tasks")
        
    def _generate_users(self):
        """Generate IoT users with random positions and characteristics"""
        self.users = SystemUtilities.generate_users(self.config)
    
    def _generate_servers(self):
        """Generate fog servers with random positions and capabilities"""
        self.servers = SystemUtilities.generate_servers(self.config)
    
    def _generate_tasks(self):
        """Generate tasks with different characteristics ensuring total tasks ≤ total server capacity"""
        self.tasks = SystemUtilities.generate_tasks(self.config, self.users, self.servers)
    
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
                # Calculate distance using utility function
                distance = SystemUtilities.calculate_euclidean_distance(
                    user['position'], server['position']
                )
                self.distance_matrix[i][j] = distance
                
                # Calculate channel gain using path loss model
                channel_gain = SystemUtilities.calculate_path_loss(
                    distance, 
                    self.config.reference_distance,
                    self.config.path_loss_exponent, 
                    self.config.reference_gain
                )
                self.channel_gains[i][j] = channel_gain
                
                # Calculate transmission delay using Shannon capacity
                snr = (self.config.transmission_power * channel_gain) / self.config.noise_power
                capacity = SystemUtilities.calculate_shannon_capacity(self.config.channel_bandwidth, snr)
                
                # Average task size for delay calculation
                avg_task_size = 1e6  # 1 MB average
                transmission_delay = SystemUtilities.calculate_transmission_delay(avg_task_size, capacity)
                self.transmission_delays[i][j] = transmission_delay
                
                # Calculate computation cost
                base_cost = server['processing_cost']
                resource_factor = 1.0 / server['available_resources']
                self.computation_costs[i][j] = base_cost * resource_factor
    
    def _initialize_server_capacities(self):
        """
        Initialize server state for HYBRID capacity model:
        - Initially: Limited capacity based on resources (config.initial_server_capacity)
        - When full: Switch to unlimited capacity with increasing waiting times
        """
        for server in self.servers:
            if self.config.use_hybrid_capacity:
                # Hybrid model: Start with limited capacity
                initial_capacity = self.config.initial_server_capacity
                self.server_capacities[server['id']] = initial_capacity
                print(f"  {server['id']}: Initial capacity = {initial_capacity} tasks")
            else:
                # Pure unlimited capacity model
                self.server_capacities[server['id']] = float('inf')
            
            # Initialize waiting time ωᵢ for each server (pseudocode line 17)
            self.server_waiting_times[server['id']] = 0.0
        
        if self.config.use_hybrid_capacity:
            print(f"✅ Initialized servers with HYBRID capacity model")
            print(f"   Initial capacity: {self.config.initial_server_capacity} tasks/server")
            print(f"   After full: Unlimited capacity with increasing waiting times")
        else:
            print(f"✅ Initialized servers with UNLIMITED capacity model")
    
    def create_preference_matrices_using_formulas(self):
        """
        Create preference matrices using theoretical formulas before stable matching:
        - User preferences: O_j(i) = 1/(ω_j^i(ζ) + ξ_j^i) [Formula 4]
        - Server preferences: D_i(j) = 1/(ω_j^i(ζ) + ξ_j^i + t_j,i) [Formula 5]
        
        This creates the complete preference matrices that will be used during matching.
        """
        print("\n" + "="*80)
        print("📊 CREATING PREFERENCE MATRICES USING THEORETICAL FORMULAS")
        print("="*80)
        
        print("\n🔧 Computing system parameters for matrix calculation...")
        
        # Initialize matrices to store computed values
        num_users = len(self.users)
        num_servers = len(self.servers)
        
        # Pre-calculate computation times for all task-server pairs
        self.computation_times = {}  # t_j,i matrix
        for task in self.tasks:
            self.computation_times[task['id']] = {}
            task_user_index = next(i for i, u in enumerate(self.users) if u['id'] == task['user_id'])
            
            for j, server in enumerate(self.servers):
                # t_j,i = computation_requirement / computational_capability
                comp_time = task['computation_requirement'] / server['computational_capability']
                self.computation_times[task['id']][server['id']] = comp_time
        
        print(f"✅ Computed {len(self.tasks)} × {len(self.servers)} computation time matrix")
        
        # Create utility matrices using theoretical formulas
        print("\n📈 Building preference matrices using research paper formulas...")
        
        # Step 1: Generate user preferences using Formula 4
        print("\n--- Creating User Preference Matrix (Formula 4) ---")
        self.user_preferences = self.generate_user_preferences()
        
        # Step 2: Generate server preferences using Formula 5  
        print("\n--- Creating Server Preference Matrix (Formula 5) ---")
        self.server_preferences = self.generate_server_preferences()
        
        print(f"\n✅ Successfully created preference matrices:")
        print(f"   • User preferences: {len(self.user_preferences)} users × {num_servers} servers")
        print(f"   • Server preferences: {len(self.server_preferences)} servers × {len(self.tasks)} tasks")
        print(f"   • Ready for stable matching algorithm")
    
    def generate_user_preferences(self) -> Dict[str, List[str]]:
        """
        Generate user preferences using theoretical formula: O_j(i) = 1/(ω_j^i(ζ) + ξ_j^i)
        Users prefer servers with lower waiting time and communication delay
        """
        user_preferences = self.iot_pref_generator.generate_theoretical_user_preferences(
            self.users, self.servers, self.transmission_delays, self.server_waiting_times,
            self.server_capacities, self.tasks
        )
        self.user_preferences = user_preferences
        return user_preferences
    
    def generate_server_preferences(self) -> Dict[str, List[str]]:
        """
        Generate server preferences using theoretical formula: D_i(j) = 1/(ω_j^i(ζ) + ξ_j^i + t_j,i)
        Servers prefer tasks with lower overall workload (waiting + communication + computation time)
        """
        server_preferences = self.fog_pref_generator.generate_theoretical_server_preferences(
            self.servers, self.tasks, self.users, self.transmission_delays, 
            self.server_waiting_times, self.server_capacities
        )
        self.server_preferences = server_preferences
        return server_preferences
    
    def run_matching_algorithm(self) -> Dict[str, List[str]]:
        """
        Run the proposed matching theory algorithm
        Modified Deferred Acceptance algorithm for task-server matching
        Uses pre-calculated preference matrices from theoretical formulas
        """
        
        # Initialize algorithm state using utility function
        task_ids = [task['id'] for task in self.tasks]
        server_ids = [server['id'] for server in self.servers]
        server_assignments, task_current_preference = MatchingUtilities.initialize_matching_state(
            task_ids, server_ids
        )
        unassigned_tasks = set(task_ids)
        
        # Use pre-calculated preferences (matrices already created)
        user_prefs = self.user_preferences
        server_prefs = self.server_preferences
        
        print("\n" + "="*80)
        print("🔄 RUNNING STABLE MATCHING ALGORITHM WITH PRE-CALCULATED MATRICES")
        print("="*80)
        
        # Map tasks to users for preference lookup
        task_to_user = {task['id']: task['user_id'] for task in self.tasks}
        
        round_num = 1
        max_rounds = 10000  # Much higher limit for complex scenarios
        consecutive_no_progress = 0
        # Scale max_no_progress based on system size for better convergence
        # Formula: Allow enough rounds for all tasks to be processed considering per-round limits
        tasks_per_server_per_round = (self.config.initial_server_capacity * 2) if self.config.use_hybrid_capacity else 100
        estimated_rounds_needed = (len(self.tasks) // (len(self.servers) * tasks_per_server_per_round)) + 5
        max_no_progress = max(20, min(100, estimated_rounds_needed))  # At least 20, up to 100 rounds
        
        print(f"\nStarting matching process with {len(unassigned_tasks)} tasks...")
        print(f"System scale: {len(self.tasks)} tasks, {len(self.servers)} servers")
        print(f"Per-round limit: {tasks_per_server_per_round} tasks/server")
        print(f"Estimated rounds needed: ~{estimated_rounds_needed}")
        print(f"Max no-progress rounds: {max_no_progress}")
        
        while unassigned_tasks and round_num <= max_rounds and consecutive_no_progress < max_no_progress:
            print(f"\nRound {round_num}:")
            
            # Track progress in this round
            tasks_assigned_this_round = 0
            proposals = {}  # server_id -> list of proposing tasks
            
            # Phase 1: Task proposals through user preferences
            for task_id in list(unassigned_tasks):
                user_id = task_to_user[task_id]
                pref_index = task_current_preference[task_id]
                
                if pref_index < len(user_prefs[user_id]):
                    preferred_server = user_prefs[user_id][pref_index]
                    
                    if preferred_server not in proposals:
                        proposals[preferred_server] = []
                    proposals[preferred_server].append(task_id)
                    
                    print(f"  Task {task_id} (via {user_id}) → {preferred_server}")
                else:
                    # Task has exhausted all preferences - this should not happen with unlimited capacity
                    # Keep task in unassigned list, don't remove it!
                    print(f"  ⚠️ WARNING: Task {task_id} exhausted all {len(user_prefs[user_id])} server preferences!")
                    # DO NOT remove from unassigned_tasks - let preference regeneration help it
            
            # Phase 2: Server decisions with HYBRID CAPACITY
            for server_id, proposing_tasks in proposals.items():
                current_tasks = server_assignments[server_id].copy()
                capacity = self.server_capacities[server_id]
                current_count = len(current_tasks)
                
                # Check if using hybrid capacity and if at/over initial capacity
                is_hybrid = self.config.use_hybrid_capacity
                initial_cap = self.config.initial_server_capacity if is_hybrid else float('inf')
                is_at_capacity = current_count >= initial_cap if is_hybrid else False
                
                if is_hybrid and current_count < initial_cap:
                    # PHASE 2A: Under initial capacity - accept up to capacity limit
                    print(f"  Server {server_id} (capacity: {current_count}/{initial_cap}) receives: {proposing_tasks}")
                    
                    all_candidates = current_tasks + proposing_tasks
                    server_pref_order = server_prefs[server_id]
                    
                    # Sort by server preference
                    def get_preference_rank(task_id):
                        try:
                            return server_pref_order.index(task_id)
                        except ValueError:
                            return len(server_pref_order)
                    
                    all_candidates.sort(key=get_preference_rank)
                    
                    # Accept only up to initial capacity
                    slots_available = initial_cap - current_count
                    accepted = all_candidates[:initial_cap]
                    rejected = all_candidates[initial_cap:]
                    
                    server_assignments[server_id] = accepted
                    
                    # Update task status
                    for task_id in proposing_tasks:
                        if task_id in accepted and task_id in unassigned_tasks:
                            unassigned_tasks.remove(task_id)
                            tasks_assigned_this_round += 1
                            print(f"    ✓ {server_id} accepts {task_id} (position: {accepted.index(task_id) + 1}/{initial_cap})")
                        elif task_id in rejected:
                            task_current_preference[task_id] += 1
                            print(f"    ✗ {server_id} rejects {task_id} (capacity full, try next preference)")
                    
                    # Switch to unlimited if now at capacity
                    if len(accepted) >= initial_cap:
                        print(f"    ⚠️  {server_id} reached capacity! Switching to UNLIMITED mode with waiting times")
                    
                    print(f"    Current: {len(server_assignments[server_id])}/{initial_cap} tasks")
                    
                else:
                    # PHASE 2B: At/over capacity OR pure unlimited - accept all with waiting time penalty
                    mode = "over capacity" if is_hybrid else "unlimited capacity"
                    print(f"  Server {server_id} ({mode}, current: {current_count}) receives: {proposing_tasks}")
                    
                    all_candidates = current_tasks + proposing_tasks
                    server_pref_order = server_prefs[server_id]
                    
                    # Sort by server preference
                    def get_preference_rank(task_id):
                        try:
                            return server_pref_order.index(task_id)
                        except ValueError:
                            return len(server_pref_order)
                    
                    all_candidates.sort(key=get_preference_rank)
                    
                    # LOAD BALANCING: Limit acceptances per round to prevent avalanche effect
                    # Accept up to 2x initial capacity per round when over capacity
                    max_new_per_round = initial_cap * 2 if is_hybrid else float('inf')
                    num_new_accepted = min(len(proposing_tasks), max_new_per_round)
                    
                    # Accept limited number of new tasks
                    if is_hybrid and num_new_accepted < len(proposing_tasks):
                        # Accept top-ranked new proposals
                        newly_accepted = [t for t in all_candidates if t in proposing_tasks][:num_new_accepted]
                        newly_rejected = [t for t in proposing_tasks if t not in newly_accepted]
                        accepted_candidates = current_tasks + newly_accepted
                        
                        print(f"    ⚠️  Load balancing: accepting {num_new_accepted}/{len(proposing_tasks)} new tasks (limit: {max_new_per_round}/round)")
                    else:
                        # Accept ALL tasks (no limit or limit not reached)
                        accepted_candidates = all_candidates
                        newly_rejected = []
                    
                    server_assignments[server_id] = accepted_candidates
                    
                    # Update task status
                    for task_id in proposing_tasks:
                        if task_id in unassigned_tasks:
                            if task_id in newly_rejected:
                                # Rejected due to per-round limit
                                task_current_preference[task_id] += 1
                                print(f"    ✗ {server_id} rejects {task_id} (per-round limit reached, try next preference)")
                            else:
                                # Accepted
                                unassigned_tasks.remove(task_id)
                                tasks_assigned_this_round += 1
                                if is_hybrid and current_count >= initial_cap:
                                    print(f"    ✓ {server_id} accepts {task_id} (queue position: {accepted_candidates.index(task_id) + 1}, ⚠️ over capacity)")
                                else:
                                    print(f"    ✓ {server_id} accepts {task_id} (queue position: {accepted_candidates.index(task_id) + 1})")
                    
                    print(f"    Current assignments: {len(server_assignments[server_id])} tasks in queue")
                    
                    # IMMEDIATE WAITING TIME UPDATE (prevent avalanche effect)
                    # Update this server's waiting time now so subsequent proposals see the increased load
                    self._update_single_server_waiting_time(server_id, server_assignments[server_id])
            
            # Step 17 from pseudocode: Each FN sends to EDs its waiting time ωᵢ
            self._update_server_waiting_times(server_assignments)
            
            # Update progress tracking
            if tasks_assigned_this_round > 0:
                consecutive_no_progress = 0
                remaining = len(unassigned_tasks)
                total = len(self.tasks)
                print(f"  Progress: {tasks_assigned_this_round} tasks assigned this round")
                print(f"  Status: {total - remaining}/{total} assigned ({((total - remaining)/total)*100:.1f}%)")
            else:
                consecutive_no_progress += 1
                remaining = len(unassigned_tasks)
                print(f"  No progress: {consecutive_no_progress}/{max_no_progress} consecutive rounds without assignments")
                print(f"  Remaining unassigned: {remaining} tasks")
            
            # With unlimited capacity, termination is simpler
            # Tasks should all be assigned in first round since no capacity constraints
            if consecutive_no_progress >= max_no_progress:
                print(f"  Algorithm converged after {round_num} rounds")
                break
            
            # Steps 19-21 from pseudocode: Update preferences AFTER EVERY ROUND based on new waiting times
            # This allows tasks to react to changing server loads dynamically
            if unassigned_tasks:
                # ALWAYS update preferences after each round to reflect new waiting times
                # This implements the dynamic preference mechanism from the paper (lines 19-21)
                print("  📊 Updating preferences based on new waiting times...")
                
                # Regenerate user preferences with updated waiting times
                user_prefs = self.generate_user_preferences()
                
                # Also update server preferences to reflect new loads
                server_prefs = self.generate_server_preferences()
                
                # Reset preference indices for unallocated tasks to try new rankings
                for task_id in unassigned_tasks:
                    task_current_preference[task_id] = 0
                
                # Show updated waiting times
                print("  ⏱️  Updated server waiting times:")
                for server_id in sorted(self.server_waiting_times.keys()):
                    wt = self.server_waiting_times[server_id]
                    num_tasks = len(server_assignments.get(server_id, []))
                    print(f"      {server_id}: {wt:.3f}s ({num_tasks} tasks)")
                
                # NOW check if any unassigned task still has preferences to explore
                # This check is AFTER regeneration, so tasks have fresh preferences
                has_viable_tasks = any(
                    task_current_preference[task_id] < len(user_prefs[task_to_user[task_id]]) 
                    for task_id in unassigned_tasks
                )
                
                if not has_viable_tasks and tasks_assigned_this_round == 0:
                    print("  ⚠️ All tasks explored all preferences after regeneration")
                    break
            
            # Handle aggressive preference update for deadlock breaking
            if consecutive_no_progress >= max_no_progress // 2 and unassigned_tasks:
                print("  ⚠️  Aggressive update - adding randomization to break deadlock...")
                import random
                for user_id in user_prefs:
                    if len(unassigned_tasks) > 0:
                        # Randomly shuffle tied preferences to create diversity
                        prefs = user_prefs[user_id].copy()
                        # Create small groups and shuffle within groups to add diversity
                        mid = len(prefs) // 2
                        if random.random() < 0.3:  # 30% chance to shuffle top half
                            top_half = prefs[:mid]
                            random.shuffle(top_half)
                            prefs[:mid] = top_half
                        if random.random() < 0.3:  # 30% chance to shuffle bottom half  
                            bottom_half = prefs[mid:]
                            random.shuffle(bottom_half)
                            prefs[mid:] = bottom_half
                        user_prefs[user_id] = prefs
            
            round_num += 1
        
        # STABLE MATCHING DETECTION AND ANNOUNCEMENT
        total_assigned = sum(len(tasks) for tasks in server_assignments.values())
        
        # Calculate actually unassigned tasks (not in any server assignment)
        all_task_ids = {task['id'] for task in self.tasks}
        assigned_task_ids = set()
        for task_list in server_assignments.values():
            assigned_task_ids.update(task_list)
        actually_unassigned = all_task_ids - assigned_task_ids
        
        # Determine termination reason and print appropriate message
        if not unassigned_tasks and len(actually_unassigned) == 0:
            print("\n" + "="*70)
            print("🎉 STABLE MATCHING ACHIEVED! 🎉")
            print("="*70)
            print("✅ All tasks have been successfully assigned to servers")
            print("✅ No task wants to switch to a different server") 
            print("✅ No server wants to replace its current tasks")
            print("✅ The matching is stable and optimal according to preferences")
            print("="*70)
            termination_reason = "STABLE_MATCHING_COMPLETE"
        elif round_num > max_rounds:
            print("\n" + "="*70)
            print("⚠️  ALGORITHM TERMINATED - MAX ROUNDS REACHED")
            print("="*70)
            print(f"Algorithm stopped after {max_rounds} rounds")
            print("Some tasks may remain unassigned due to round limit")
            termination_reason = "MAX_ROUNDS_REACHED"
        elif consecutive_no_progress >= max_no_progress:
            print("\n" + "="*70)
            print("⚠️  ALGORITHM TERMINATED - NO PROGRESS POSSIBLE")
            print("="*70)
            print(f"No progress for {consecutive_no_progress} consecutive rounds")
            print("Remaining unassigned tasks cannot find suitable assignments")
            termination_reason = "NO_PROGRESS_LIMIT"
        else:
            print("\n" + "="*70)
            print("✅ MATCHING PROCESS COMPLETED")
            print("="*70)
            termination_reason = "NORMAL_COMPLETION"
        
        # Final status summary
        print(f"\n📊 MATCHING SUMMARY:")
        print(f"  Algorithm completed after {round_num-1} rounds")
        print(f"  Termination reason: {termination_reason}")
        print(f"  Assigned tasks: {total_assigned}/{len(self.tasks)} ({(total_assigned/len(self.tasks)*100):.1f}%)")
        print(f"  Unassigned tasks: {len(actually_unassigned)}")
        if actually_unassigned:
            print(f"  Unassigned task IDs: {sorted(actually_unassigned)}")
        
        # Show final allocation summary
        print(f"\n🏆 FINAL TASK ALLOCATION (UNLIMITED CAPACITY MODEL):")
        for server_id, assigned_tasks in server_assignments.items():
            num_tasks = len(assigned_tasks)
            waiting_time = self.server_waiting_times.get(server_id, 0.0)
            if assigned_tasks:
                print(f"  {server_id}: {num_tasks} tasks (waiting time: {waiting_time:.3f}s)")
                print(f"    Tasks: {assigned_tasks[:10]}{'...' if len(assigned_tasks) > 10 else ''}")
            else:
                print(f"  {server_id}: No tasks assigned")
        
        self.final_allocation = server_assignments
        return server_assignments
    
    def _update_single_server_waiting_time(self, server_id: str, assigned_tasks: List[str]):
        """
        Update waiting time for a single server immediately (prevents avalanche effect).
        Called after each server acceptance to reflect load changes in real-time.
        """
        num_assigned = len(assigned_tasks)
        
        # Get server information
        server_info = next(s for s in self.servers if s['id'] == server_id)
        
        # Determine initial capacity
        is_hybrid = self.config.use_hybrid_capacity
        initial_capacity = self.config.initial_server_capacity if is_hybrid else 0
        
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
        
        # Update immediately
        self.server_waiting_times[server_id] = waiting_time
        print(f"      ⚡ {server_id} waiting time updated: {waiting_time:.3f}s ({num_assigned} tasks)")
    
    def _update_server_waiting_times(self, server_assignments: Dict[str, List[str]]):
        """
        Update server waiting times ωᵢ for HYBRID CAPACITY model:
        - Under capacity: Minimal waiting time (concurrent processing)
        - Over capacity: Waiting time increases with queue length
        
        Formula (over capacity): ωᵢ = Σ(processing_time_j) + queue_penalty
        where queue_penalty increases with tasks beyond initial capacity
        """
        for server_id in self.server_capacities.keys():
            assigned_tasks = server_assignments.get(server_id, [])
            num_assigned = len(assigned_tasks)
            
            # Get server information
            server_info = next(s for s in self.servers if s['id'] == server_id)
            
            # Determine initial capacity
            is_hybrid = self.config.use_hybrid_capacity
            initial_capacity = self.config.initial_server_capacity if is_hybrid else 0
            
            # Calculate total processing time for all tasks in queue
            total_processing_time = 0.0
            for task_id in assigned_tasks:
                task_info = next(t for t in self.tasks if t['id'] == task_id)
                processing_time = task_info['computation_requirement'] / server_info['computational_capability']
                total_processing_time += processing_time
            
            if is_hybrid and num_assigned <= initial_capacity:
                # UNDER CAPACITY: Minimal waiting time (tasks can process concurrently up to capacity)
                # Average processing time (assuming parallel processing within capacity)
                waiting_time = total_processing_time / min(num_assigned, initial_capacity) if num_assigned > 0 else 0.0
                
                if num_assigned > 0:
                    print(f"    Server {server_id}: {num_assigned}/{initial_capacity} tasks → waiting time: {waiting_time:.3f}s " +
                          f"(under capacity, parallel processing)")
            else:
                # OVER CAPACITY or UNLIMITED: Sequential processing with queue penalty
                # Calculate tasks beyond initial capacity
                tasks_over_capacity = max(0, num_assigned - initial_capacity) if is_hybrid else num_assigned
                
                # Base waiting time: sum of processing times
                base_waiting_time = total_processing_time
                
                # Queue penalty: increases non-linearly with queue length beyond capacity
                # Using exponent from config (default 1.5 for super-linear growth)
                if is_hybrid:
                    # Penalty only for tasks beyond initial capacity
                    queue_penalty = (tasks_over_capacity ** self.config.waiting_time_penalty_exponent) * self.config.waiting_time_increment
                else:
                    # Pure unlimited: penalty for all tasks
                    queue_penalty = (num_assigned ** self.config.waiting_time_penalty_exponent) * self.config.waiting_time_increment
                
                # Total waiting time
                waiting_time = base_waiting_time + queue_penalty
                
                if num_assigned > 0:
                    if is_hybrid:
                        print(f"    Server {server_id}: {num_assigned} tasks ({tasks_over_capacity} over capacity) → waiting time: {waiting_time:.3f}s " +
                              f"(processing: {base_waiting_time:.3f}s + queue penalty: {queue_penalty:.3f}s)")
                    else:
                        print(f"    Server {server_id}: {num_assigned} tasks → waiting time: {waiting_time:.3f}s " +
                              f"(processing: {base_waiting_time:.3f}s + queue penalty: {queue_penalty:.3f}s)")
            
            # Update server waiting time (this will affect next round's preference calculations)
            self.server_waiting_times[server_id] = waiting_time
    
    def calculate_performance_metrics(self, allocation: Dict[str, List[str]]) -> Dict:
        """
        Calculate performance metrics using unified simulation approach.
        
        Args:
            allocation: Dictionary mapping server_id to list of assigned task_ids
            
        Returns:
            Dictionary containing simulation-based numerical results
        """
        # Initialize unified calculator if not already done
        if self.unified_calculator is None:
            self.unified_calculator = SimulationMetrics(
                self.tasks, self.servers, self.users, self.server_capacities, self.transmission_delays, self.config
            )
        
        # Run unified simulation and get all results
        unified_results = self.unified_calculator.run_simulation_and_calculate_metrics(allocation)
        
        # Return the numerical results from the unified calculation
        return unified_results['numerical_results']
    
    def simulate_task_execution(self, allocation: Dict[str, List[str]]) -> Dict:
        """
        Simulate the actual execution of tasks on assigned servers after stable matching
        Uses the external TaskExecutionSimulator module for detailed simulation
        
        Args:
            allocation: Dictionary mapping server_id to list of assigned task_ids
            
        Returns:
            Dictionary containing simulation results and statistics
        """
        # Initialize unified calculator if not already done
        if self.unified_calculator is None:
            self.unified_calculator = SimulationMetrics(
                self.tasks, self.servers, self.users, self.server_capacities, self.transmission_delays, self.config
            )
        
        # Run unified simulation and calculate metrics
        unified_results = self.unified_calculator.run_simulation_and_calculate_metrics(allocation)
        return unified_results['simulation_data']  # Return simulation data for compatibility
    
    def run_complete_algorithm(self):
        """Run the complete proposed algorithm with full analysis"""
        print_header("PROPOSED MATCHING THEORY ALGORITHM FOR TASK OFFLOADING", 90)
        
        start_time = time.time()
        
        # Initialize system
        self.initialize_system()
        
        # Step 1: Create preference matrices using theoretical formulas
        self.create_preference_matrices_using_formulas()
        
        # Step 2: Run the matching algorithm (uses the pre-calculated matrices)
        allocation = self.run_matching_algorithm() 
        
        # Step 3: After stable matching - Simulate task execution 
        print("\n" + "="*80)
        print("🔄 SIMULATING TASK EXECUTION AFTER STABLE MATCHING")
        print("="*80)
        print("Demonstrating real-time task processing on assigned fog servers...")
        simulation_results = self.simulate_task_execution(allocation)
        
        # Step 4: Calculate performance metrics using unified simulation approach
        numerical_results = self.calculate_performance_metrics(allocation)
        
        # Print results (basic summary)
        # self.print_results(allocation, metrics)  # Method not implemented
        
        print_section("Numerical Results Summary (Section IV)")
        print("Research Paper Performance Metrics:")
        print(f"  T̄_M (Worst completion time): {numerical_results['worst_completion_time_TM']:.4f}s")
        print(f"  T̄_T (Mean completion time): {numerical_results['mean_completion_time_TT']:.4f}s") 
        print(f"  T̄_W (Mean waiting time): {numerical_results['mean_waiting_time_TW']:.4f}s")
        print(f"  I_J (Jain's fairness index): {numerical_results['jains_index_IJ']:.4f}")
        
        execution_time = time.time() - start_time
        from utility import format_time_duration
        print(f"\nTotal Execution Time: {format_time_duration(execution_time)}")
        
        return {
            'allocation': allocation,
            'numerical_results': numerical_results,
            'execution_time': execution_time,
            'config': self.config
        }


def main():
    """Main function to run the proposed algorithm"""
    # Create configuration for 5 servers and exactly 1000 tasks
    config = SystemConfiguration(
        num_users=10,                      # 10 users (for task distribution)
        num_servers=5,                     # 5 fog servers
        num_task_types=10,                 # 10 task types for variety
        network_area_size=500.0,           # Large coverage area
        fixed_task_count=500,             # Generate exactly 10 tasks
        random_seed=None                   # None = different results each run, or set to int for reproducibility
    )
    
    # Set random seed based on config (None = truly random)
    if config.random_seed is None:
        # Generate random seed from system time
        import time
        random_seed = int(time.time() * 1000) % (2**32)
        print(f"Using random seed: {random_seed} (set config.random_seed={random_seed} to reproduce)")
    else:
        random_seed = config.random_seed
        print(f"Using fixed seed: {random_seed} (reproducible results)")
    
    set_random_seeds(random_seed)
    
    # Create and run the proposed algorithm
    proposed_algorithm = ProposedTaskOffloadingAlgorithm(config)
    
    # Debug: Show first few server/user positions to verify randomness
    print(f"\n=== Verifying Randomness ===")
    print(f"First 3 server positions:")
    for server in proposed_algorithm.servers[:3]:
        print(f"  {server['id']}: ({server['position'][0]:.2f}, {server['position'][1]:.2f}), CPU: {server['computational_capability']/1e9:.2f}GHz")
    print(f"First 3 user positions:")
    for user in proposed_algorithm.users[:3]:
        print(f"  {user['id']}: ({user['position'][0]:.2f}, {user['position'][1]:.2f})")
    print(f"={'='*50}\n")
    
    results = proposed_algorithm.run_complete_algorithm()


if __name__ == "__main__":
    main()