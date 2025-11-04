"""
Simulation Utilities
Helper functions specifically for the Simulation module
"""

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
