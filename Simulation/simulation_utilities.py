"""
Simulation Utilities
Helper functions specifically for the Simulation module
Includes realistic randomness functions for task execution simulation
"""

import random
import numpy as np
from typing import Dict, List, Any


class SimulationUtilities:
    """Utility functions for simulation calculations"""
    
    @staticmethod
    def calculate_jains_fairness_index(values: List[float]) -> float:
        """
        Calculate Jain's Fairness Index
        
        Args:
            values: List of values (e.g., completion times, utilizations)
            
        Returns:
            Fairness index (0 to 1, where 1 is perfectly fair)
        """
        if not values or len(values) == 0:
            return 0.0
        
        if len(values) == 1:
            return 1.0
        
        n = len(values)
        sum_v = sum(values)
        sum_v_squared = sum(v*v for v in values)
        
        if sum_v_squared > 0:
            return (sum_v * sum_v) / (n * sum_v_squared)
        else:
            return 1.0
    
    @staticmethod
    def calculate_mean(values: List[float]) -> float:
        """Calculate mean of values"""
        return sum(values) / len(values) if values else 0.0
    
    @staticmethod
    def calculate_std_deviation(values: List[float]) -> float:
        """Calculate standard deviation"""
        if not values or len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5
    
    @staticmethod
    def calculate_utilization(processing_time: float, total_time: float) -> float:
        """
        Calculate server utilization percentage
        
        Args:
            processing_time: Total time spent processing
            total_time: Total simulation time
            
        Returns:
            Utilization percentage (0-100)
        """
        if total_time <= 0:
            return 0.0
        return min(processing_time / total_time * 100, 100.0)
    
    @staticmethod
    def add_processing_time_variation(base_time: float, variation_percent: float = 15.0) -> float:
        """
        Add realistic random variation to processing time to simulate real-world CPU effects
        
        Simulates:
        - CPU scheduling variations
        - Cache hits/misses
        - Temperature throttling
        - Memory contention
        - OS context switching
        - I/O interrupts
        
        Args:
            base_time: Theoretical base processing time (seconds)
            variation_percent: Percentage of variation (±variation_percent)
            
        Returns:
            Realistic processing time with random variation
        """
        # Calculate variation range
        variation_range = base_time * (variation_percent / 100.0)
        
        # Add random variation (can be positive or negative)
        # Use uniform distribution for realistic CPU behavior
        realistic_time = base_time + random.uniform(-variation_range, variation_range)
        
        # Ensure result is never negative
        return max(realistic_time, base_time * 0.01)
    
    @staticmethod
    def add_startup_delay(min_delay: float = 0.002, max_delay: float = 0.015) -> float:
        """
        Add random startup delay to simulate task initialization overhead
        
        Simulates:
        - Context switching overhead
        - Task initialization
        - Cache warming (loading data into cache)
        - Driver/system call latency
        - Memory allocation
        - Resource setup
        
        Args:
            min_delay: Minimum startup delay in seconds (default 2ms)
            max_delay: Maximum startup delay in seconds (default 15ms)
            
        Returns:
            Random startup delay between min_delay and max_delay
        """
        # Use uniform distribution for realistic startup variations
        return random.uniform(min_delay, max_delay)
    
    @staticmethod
    def add_network_jitter(base_delay: float, jitter_percent: float = 10.0) -> float:
        """
        Add realistic network jitter to transmission delays
        
        Simulates:
        - Packet retransmissions
        - Network congestion
        - Route fluctuations
        - Buffer variations
        
        Args:
            base_delay: Base transmission delay (seconds)
            jitter_percent: Maximum jitter as percentage of base delay
            
        Returns:
            Delay with realistic network jitter added
        """
        jitter_range = base_delay * (jitter_percent / 100.0)
        jittered_delay = base_delay + random.uniform(-jitter_range, jitter_range)
        return max(jittered_delay, base_delay * 0.01)
    
    @staticmethod
    def add_queue_wait_variation(base_wait: float, load_factor: float = 1.0) -> float:
        """
        Add realistic variation to queue waiting times based on system load
        
        Simulates:
        - Queue scheduling variations
        - Priority-based preemption
        - Bursty task arrivals
        - Load-dependent delays
        
        Args:
            base_wait: Base waiting time (seconds)
            load_factor: System load factor (1.0 = normal, >1.0 = high load)
            
        Returns:
            Realistic queue waiting time with variation
        """
        # Higher variation when system is more loaded
        variation_percent = 20.0 * load_factor
        variation_range = base_wait * (variation_percent / 100.0)
        
        # Use exponential distribution for queue wait (more realistic than uniform)
        # But with controlled bounds
        varied_wait = base_wait + random.uniform(-variation_range, variation_range * 2)
        return max(varied_wait, 0.0)
    
    @staticmethod
    def simulate_task_interference(base_time: float, concurrent_tasks: int = 1) -> float:
        """
        Simulate the impact of task interference/contention
        
        When multiple tasks run concurrently on same server:
        - Cache conflicts
        - Memory bandwidth sharing
        - CPU core contention
        - Bus contention
        
        Args:
            base_time: Original task time (seconds)
            concurrent_tasks: Number of tasks running concurrently
            
        Returns:
            Adjusted time accounting for interference
        """
        if concurrent_tasks <= 1:
            return base_time
        
        # Each additional concurrent task adds overhead
        # Typical overhead: 10-30% per additional task
        interference_factor = 1.0 + (concurrent_tasks - 1) * random.uniform(0.1, 0.3)
        
        # Add some randomness to the interference
        noise = random.uniform(0.95, 1.05)
        
        return base_time * interference_factor * noise
    
    @staticmethod
    def generate_task_arrival_time(max_arrival_window: float = 5.0) -> float:
        """
        Generate realistic task arrival time to simulate asynchronous IoT task submission
        
        In real IoT systems, tasks don't all arrive at exactly time 0:
        - IoT devices detect events at different times
        - Network delays cause submission time variations
        - Device wake-up schedules differ
        - Sensor trigger times are asynchronous
        
        Args:
            max_arrival_window: Maximum time window for task arrivals (seconds)
                               Default 5.0s means tasks can arrive within first 5 seconds
        
        Returns:
            Random task arrival time between 0 and max_arrival_window (seconds)
        """
        # Use exponential distribution for more realistic arrival patterns
        # (Most tasks arrive early, fewer arrive later - typical of IoT burst patterns)
        # Lambda = 1/2 gives good spread across the window
        lambda_param = 2.0 / max_arrival_window
        arrival_time = random.expovariate(lambda_param)
        
        # Cap at max_arrival_window to ensure tasks arrive within reasonable time
        return min(arrival_time, max_arrival_window)


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
