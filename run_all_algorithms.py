import sys
import os
import time
from typing import Dict, List, Any
import pandas as pd

# Add the current directory to sys.path to ensure imports work correctly
sys.path.append(os.getcwd())

from config import SystemConfiguration
from Algorithm.algorithm import ProposedTaskOffloadingAlgorithm
from Algorithm.greedy_algorithm import GreedyTaskOffloadingAlgorithm
from Algorithm.random_algorithm import RandomTaskOffloadingAlgorithm
from Algorithm.simple_matching_algorithm import ProposedTaskOffloadingAlgorithm as SimpleMatchingAlgorithm
from Algorithm.algorithm_utilities import set_random_seeds

def format_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Extract and format key metrics for comparison"""
    return {
        "Avg Completion Time (s)": f"{metrics.get('mean_completion_time_TT', 0):.4f}",
        "Avg Waiting Time (s)": f"{metrics.get('mean_waiting_time_TW', 0):.4f}",
        "Worst Completion Time (s)": f"{metrics.get('worst_completion_time_TM', 0):.4f}",
        "Jain's Fairness Index": f"{metrics.get('jains_index_IJ', 0):.4f}",
        "Avg Server Utilization (%)": f"{metrics.get('avg_server_utilization', 0):.2f}",
        "Task Success Rate (%)": f"{metrics.get('assignment_success_rate', 0)*100:.1f}"
    }

import contextlib

@contextlib.contextmanager
def suppress_output():
    """Context manager to suppress stdout"""
    with open(os.devnull, 'w', encoding='utf-8') as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

def run_simulation():
    print("=" * 100)
    print("RUNNING COMPARATIVE SIMULATION: 4 ALGORITHMS, 1000 TASKS")
    print("=" * 100)
    
    # Create configuration
    # Note: Using num_users=10 to match your manual runs, but with fixed seed for fairness
    config = SystemConfiguration(
        num_users=10,                      # 10 users (matching your manual run)
        num_servers=5, 
        num_task_types=10,
        network_area_size=500.0,
        fixed_task_count=1000,  # Explicitly set to 1000 tasks
        random_seed=42,        # Fixed seed for FAIR comparison
        use_multilevel=True,
        local_processing_threshold=2.0
    )
    
    results_summary = []
    
    # 1. Proposed Algorithm (Matching Theory with Hierarchy)
    print("\n\n" + "-"*50)
    print("1. PROPOSED ALGORITHM (Hierarchical Matching Theory)")
    print("-" * 50)
    print("Running simulation... (Output suppressed for performance)")
    
    with suppress_output():
        set_random_seeds(42)
        proposed_algo = ProposedTaskOffloadingAlgorithm(config)
        proposed_algo.initialize_system()
        
        # Need to manually run the steps for proposed algorithm as it doesn't have a 'run_complete_algorithm'
        # capable of returning metrics directly in one call like the others might, 
        # but let's check what methods are available. 
        # Algorithm.py has `run_matching_algorithm` returning allocation.
        
        proposed_algo.create_preference_matrices_using_formulas()
        proposed_allocation = proposed_algo.run_matching_algorithm()
        
        # Calculate metrics
        if proposed_algo.unified_calculator is None:
            from Simulation.simulation_metrics import SimulationMetrics
            proposed_algo.unified_calculator = SimulationMetrics(
                proposed_algo.tasks, proposed_algo.servers, proposed_algo.users, 
                proposed_algo.server_capacities, proposed_algo.transmission_delays, config
            )
        
        proposed_metrics_full = proposed_algo.unified_calculator.run_simulation_and_calculate_metrics(proposed_allocation)
        proposed_metrics = proposed_metrics_full['numerical_results']
    
    results_summary.append({
        "Algorithm": "Proposed (Hierarchical)",
        **format_metrics(proposed_metrics)
    })
    print("Done.")

    # 2. Greedy Algorithm
    print("\n\n" + "-"*50)
    print("2. GREEDY ALGORITHM")
    print("-" * 50)
    print("Running simulation... (Output suppressed for performance)")
    
    with suppress_output():
        set_random_seeds(42) # Reset seed for fairness
        greedy_algo = GreedyTaskOffloadingAlgorithm(config)
        greedy_results = greedy_algo.run_complete_algorithm()
    
    results_summary.append({
        "Algorithm": "Greedy",
        **format_metrics(greedy_results['numerical_results'])
    })
    print("Done.")

    # 3. Random Algorithm
    print("\n\n" + "-"*50)
    print("3. RANDOM ALGORITHM")
    print("-" * 50)
    print("Running simulation... (Output suppressed for performance)")
    
    with suppress_output():
        set_random_seeds(42)
        random_algo = RandomTaskOffloadingAlgorithm(config)
        random_results = random_algo.run_complete_algorithm()
    
    results_summary.append({
        "Algorithm": "Random",
        **format_metrics(random_results['numerical_results'])
    })
    print("Done.")

    # 4. Simple Matching Algorithm (Single Level)
    print("\n\n" + "-"*50)
    print("4. SIMPLE MATCHING ALGORITHM (Flat Architecture)")
    print("-" * 50)
    print("Running simulation... (Output suppressed for performance)")
    
    with suppress_output():
        set_random_seeds(42)
        # Simple algorithm is FLAT architecture (no hierarchy)
        # We need to explicitly disable multi-level for it to match individual run
        simple_config = SystemConfiguration(
            num_users=config.num_users,
            num_servers=config.num_servers,
            num_task_types=config.num_task_types,
            network_area_size=config.network_area_size,
            fixed_task_count=config.fixed_task_count,
            random_seed=config.random_seed,
            use_multilevel=False,  # FORCE SINGLE LEVEL for Simple Algorithm
            local_processing_threshold=config.local_processing_threshold
        )
        simple_algo = SimpleMatchingAlgorithm(simple_config)
        simple_algo.initialize_system()
        simple_algo.create_preference_matrices_using_formulas()
        simple_allocation = simple_algo.run_matching_algorithm()
        
        # Calculate metrics for simple algorithm
        if simple_algo.unified_calculator is None:
            from Simulation.simulation_metrics import SimulationMetrics
            simple_algo.unified_calculator = SimulationMetrics(
                simple_algo.tasks, simple_algo.servers, simple_algo.users, 
                simple_algo.server_capacities, simple_algo.transmission_delays, config
            )
            
        simple_metrics_full = simple_algo.unified_calculator.run_simulation_and_calculate_metrics(simple_allocation)
        simple_metrics = simple_metrics_full['numerical_results']
    
    results_summary.append({
        "Algorithm": "Simple (Flat)",
        **format_metrics(simple_metrics)
    })
    print("Done.")
    
    # ==========================================
    # 📦 RESULTS DISPLAY (2 BOXES)
    # ==========================================
    
    # Helper to print ASCII box table
    def print_box_table(title, columns, data, alignments=None):
        if not data: return
        
        # Calculate column widths
        col_widths = [len(c) for c in columns]
        for row in data:
            for i, val in enumerate(row):
                col_widths[i] = max(col_widths[i], len(str(val)))
        
        # Add padding
        col_widths = [w + 2 for w in col_widths]
        
        # Build separators
        top_border = "┌" + "┬".join("─" * w for w in col_widths) + "┐"
        header_sep = "├" + "┼".join("─" * w for w in col_widths) + "┤"
        bottom_border = "└" + "┴".join("─" * w for w in col_widths) + "┘"
        
        print(f"\n{title}")
        print(top_border)
        
        # Print Header
        header_row = "│"
        for i, col in enumerate(columns):
            header_row += f" {col:<{col_widths[i]-1}}│"
        print(header_row)
        print(header_sep)
        
        # Print Rows
        for row in data:
            data_row = "│"
            for i, val in enumerate(row):
                align = "<"
                if alignments and i < len(alignments):
                    align = alignments[i]
                data_row += f" {str(val):{align}{col_widths[i]-1}}│"
            print(data_row)
            
        print(bottom_border)

    # ------------------------------------------
    # BOX 1: Performance Metrics (Matching Image)
    # ------------------------------------------
    box1_columns = ["Algorithm", "Avg Completion", "Avg Waiting", "Worst Completion", "Fairness Index"]
    box1_data = []
    
    for res in results_summary:
        box1_data.append([
            res["Algorithm"],
            res["Avg Completion Time (s)"],
            res["Avg Waiting Time (s)"],
            res["Worst Completion Time (s)"],
            res["Jain's Fairness Index"]
        ])
        
    print_box_table(
        "📊 PERFORMANCE METRICS", 
        box1_columns, 
        box1_data,
        alignments=["<", ">", ">", ">", ">"]
    )

    # ------------------------------------------
    # BOX 2: System Capabilities & Efficiency
    # ------------------------------------------
    box2_columns = ["Algorithm", "Server Utilization", "Success Rate"]
    box2_data = []
    
    for res in results_summary:
        box2_data.append([
            res["Algorithm"],
            f"{res['Avg Server Utilization (%)']}%",
            f"{res['Task Success Rate (%)']}%"
        ])
        
    print_box_table(
        "🏭 SYSTEM EFFICIENCY", 
        box2_columns, 
        box2_data,
        alignments=["<", ">", ">"]
    )
    
    # Verification
    print("\n✅ Verification:")
    print(f"Task Count: {config.fixed_task_count}")
    print(f"Random Seed: {config.random_seed}")

if __name__ == "__main__":
    run_simulation()
