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
    from Simulation.simulation_metrics import SimulationMetrics
    NUM_RUNS = 10
    print("=" * 100)
    print(f"RUNNING MONTE CARLO SIMULATION: 4 ALGORITHMS, 1000 TASKS, {NUM_RUNS} ITERATIONS")
    print("=" * 100)
    
    # Storage for aggregating results across runs
    # Structure: {'AlgorithmName': {'metric': [values]}}
    aggregated_results = {
        'Proposed (Hierarchical)': {'completion': [], 'waiting': [], 'worst': [], 'fairness': [], 'utilization': [], 'success': []},
        'Greedy': {'completion': [], 'waiting': [], 'worst': [], 'fairness': [], 'utilization': [], 'success': []},
        'Random': {'completion': [], 'waiting': [], 'worst': [], 'fairness': [], 'utilization': [], 'success': []},
        'Simple (Flat)': {'completion': [], 'waiting': [], 'worst': [], 'fairness': [], 'utilization': [], 'success': []}
    }

    for run_idx in range(NUM_RUNS):
        current_seed = 42 + run_idx
        print(f"\n🔄 Iteration {run_idx + 1}/{NUM_RUNS} (Seed: {current_seed})")
        print("-" * 50)

        # Create configuration for this run
        config = SystemConfiguration(
            num_users=10,
            num_servers=5, 
            num_task_types=10,
            network_area_size=500.0,
            fixed_task_count=1000,
            random_seed=current_seed,  # Vary seed per run
            use_multilevel=True,
            local_processing_threshold=2.0
        )
        
        # 1. Proposed Algorithm
        print("   Running Proposed Algorithm...", end="", flush=True)
        with suppress_output():
            set_random_seeds(current_seed)
            proposed_algo = ProposedTaskOffloadingAlgorithm(config)
            proposed_algo.initialize_system()
            proposed_algo.create_preference_matrices_using_formulas()
            proposed_allocation = proposed_algo.run_matching_algorithm()
            
            if proposed_algo.unified_calculator is None:
                calc = SimulationMetrics(proposed_algo.tasks, proposed_algo.servers, proposed_algo.users, 
                                        proposed_algo.server_capacities, proposed_algo.transmission_delays, config)
                proposed_algo.unified_calculator = calc
            
            p_metrics = proposed_algo.unified_calculator.run_simulation_and_calculate_metrics(proposed_allocation)
            
            # Extract basic metrics from numerical_results
            p_res = p_metrics['numerical_results']
            stats = _extract_metrics(p_res)
            _store_metrics(aggregated_results['Proposed (Hierarchical)'], stats)
        print(" Done.")

        # 2. Greedy Algorithm
        print("   Running Greedy Algorithm...", end="", flush=True)
        with suppress_output():
            set_random_seeds(current_seed)
            greedy_algo = GreedyTaskOffloadingAlgorithm(config)
            greedy_algo.initialize_system()
            greedy_results = greedy_algo.run_complete_algorithm()
            
            g_res = greedy_results['numerical_results']
            stats = _extract_metrics(g_res)
            _store_metrics(aggregated_results['Greedy'], stats)
        print(" Done.")

        # 3. Random Algorithm
        print("   Running Random Algorithm...", end="", flush=True)
        with suppress_output():
            set_random_seeds(current_seed)
            random_algo = RandomTaskOffloadingAlgorithm(config)
            random_algo.initialize_system()
            random_results = random_algo.run_complete_algorithm()
            
            r_res = random_results['numerical_results']
            stats = _extract_metrics(r_res)
            _store_metrics(aggregated_results['Random'], stats)
        print(" Done.")

        # 4. Simple Algorithm (WITH CLOUD ACCESS)
        print("   Running Simple Algorithm...", end="", flush=True)
        with suppress_output():
            set_random_seeds(current_seed)
            simple_config = SystemConfiguration(
                num_users=config.num_users,
                num_servers=config.num_servers,
                num_task_types=config.num_task_types,
                network_area_size=config.network_area_size,
                fixed_task_count=config.fixed_task_count,
                random_seed=current_seed,
                use_multilevel=True,  # KEEP TRUE FOR FAIRNESS
                local_processing_threshold=config.local_processing_threshold
            )
            simple_algo = SimpleMatchingAlgorithm(simple_config)
            simple_algo.initialize_system()
            simple_algo.create_preference_matrices_using_formulas()
            simple_allocation = simple_algo.run_matching_algorithm()
            
            if simple_algo.unified_calculator is None:
                calc = SimulationMetrics(simple_algo.tasks, simple_algo.servers, simple_algo.users, 
                                        simple_algo.server_capacities, simple_algo.transmission_delays, simple_config)
                simple_algo.unified_calculator = calc
            
            s_metrics = simple_algo.unified_calculator.run_simulation_and_calculate_metrics(simple_allocation)
            s_res = s_metrics['numerical_results']
            
            stats = _extract_metrics(s_res)
            _store_metrics(aggregated_results['Simple (Flat)'], stats)
        print(" Done.")

    # ==========================================
    # 📦 AVERAGED RESULTS DISPLAY
    # ==========================================
    
    print("\n\n" + "="*100)
    print(f"✅ MONTE CARLO SIMULATION COMPLETE ({NUM_RUNS} RUNS)")
    print("="*100)
    
    # Calculate Averages
    final_summary = []
    for algo_name, metrics in aggregated_results.items():
        if not metrics['completion']: continue # Skip if empty
        
        final_summary.append({
            "Algorithm": algo_name,
            "Avg Completion": sum(metrics['completion']) / NUM_RUNS,
            "Avg Waiting": sum(metrics['waiting']) / NUM_RUNS,
            "Worst Completion": sum(metrics['worst']) / NUM_RUNS,
            "Fairness Index": sum(metrics['fairness']) / NUM_RUNS,
            "Server Utilization": sum(metrics['utilization']) / NUM_RUNS,
            "Success Rate": sum(metrics['success']) / NUM_RUNS
        })
    
    # Print Tables
    # ------------------------------------------
    # BOX 1: Performance Metrics
    # ------------------------------------------
    box1_columns = ["Algorithm", "Avg Completion", "Avg Waiting", "Worst Completion", "Fairness Index"]
    box1_data = []
    
    for res in final_summary:
        box1_data.append([
            res["Algorithm"],
            f"{res['Avg Completion']:.4f}",
            f"{res['Avg Waiting']:.4f}",
            f"{res['Worst Completion']:.4f}",
            f"{res['Fairness Index']:.4f}"
        ])
        
    print_box_table(
        "📊 AVERAGED PERFORMANCE METRICS (10 RUNS)", 
        box1_columns, 
        box1_data,
        alignments=["<", ">", ">", ">", ">"]
    )

    # ------------------------------------------
    # BOX 2: System Efficiency
    # ------------------------------------------
    box2_columns = ["Algorithm", "Server Utilization", "Success Rate"]
    box2_data = []
    
    for res in final_summary:
        box2_data.append([
            res["Algorithm"],
            f"{res['Server Utilization']:.2f}%",
            f"{res['Success Rate']:.1f}%"
        ])
        
    print_box_table(
        "🏭 AVERAGED SYSTEM EFFICIENCY", 
        box2_columns, 
        box2_data,
        alignments=["<", ">", ">"]
    )

# Helper to extract metrics from SimulationMetrics outputs
def _extract_metrics(results_dict):
    # results_dict is 'numerical_results' dict
    
    avg_comp = results_dict.get('mean_completion_time_TT', 0)
    avg_wait = results_dict.get('mean_waiting_time_TW', 0)
    worst_comp = results_dict.get('worst_completion_time_TM', 0)
    jains = results_dict.get('jains_index_IJ', 0)
    avg_util = results_dict.get('avg_server_utilization', 0)
    success_rate = results_dict.get('assignment_success_rate', 0) * 100
    
    return {
        'completion': avg_comp,
        'waiting': avg_wait,
        'worst': worst_comp,
        'fairness': jains,
        'utilization': avg_util,
        'success': success_rate
    }

def _store_metrics(storage, stats):
    storage['completion'].append(stats['completion'])
    storage['waiting'].append(stats['waiting'])
    storage['worst'].append(stats['worst'])
    storage['fairness'].append(stats['fairness'])
    storage['utilization'].append(stats['utilization'])
    storage['success'].append(stats['success'])

def print_box_table(title, columns, data, alignments=None):
    if not data: return
    col_widths = [len(c) for c in columns]
    for row in data:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(val)))
    col_widths = [w + 2 for w in col_widths]
    top_border = "┌" + "┬".join("─" * w for w in col_widths) + "┐"
    header_sep = "├" + "┼".join("─" * w for w in col_widths) + "┤"
    bottom_border = "└" + "┴".join("─" * w for w in col_widths) + "┘"
    print(f"\n{title}")
    print(top_border)
    header_row = "│"
    for i, col in enumerate(columns):
        header_row += f" {col:<{col_widths[i]-1}}│"
    print(header_row)
    print(header_sep)
    for row in data:
        data_row = "│"
        for i, val in enumerate(row):
            align = "<"
            if alignments and i < len(alignments): align = alignments[i]
            data_row += f" {str(val):{align}{col_widths[i]-1}}│"
        print(data_row)
    print(bottom_border)

if __name__ == "__main__":
    run_simulation()
