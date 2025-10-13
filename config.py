"""
System Configuration for Matching Theory Task Offloading Algorithm
Contains configuration parameters and settings for the proposed algorithm
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class SystemConfiguration:
    """Configuration parameters for the proposed algorithm"""
    
    # System topology parameters
    num_users: int = 5                    # Number of IoT users  
    num_servers: int = 4                  # Number of fog servers
    num_task_types: int = 4               # Number of task types
    max_server_capacity: int = 4          # Maximum tasks per server (auto-calculated if fixed_task_count set)
    network_area_size: float = 100.0      # Deployment area size (meters)
    
    # Wireless communication parameters
    transmission_power: float = 0.1       # Transmission power (Watts)
    channel_bandwidth: float = 1e6        # Channel bandwidth (Hz)
    noise_power: float = 1e-12           # Noise power (Watts)
    path_loss_exponent: float = 2.0      # Path loss exponent
    reference_distance: float = 1.0       # Reference distance (meters)
    reference_gain: float = 1e-3         # Reference channel gain
    
    # Computation parameters
    computation_density: float = 1000     # CPU cycles per bit
    server_frequency: float = 2e9        # Server CPU frequency (Hz)
    
    # Task parameters
    min_task_data_size: float = 1e6      # Minimum task data size (bits)
    max_task_data_size: float = 5e6      # Maximum task data size (bits)
    min_task_priority: int = 1           # Minimum task priority
    max_task_priority: int = 5           # Maximum task priority
    fixed_task_count: Optional[int] = None  # Fixed number of tasks to generate (overrides user-based generation)
    
    def __post_init__(self):
        """Auto-calculate server capacity based on fixed task count"""
        if self.fixed_task_count is not None and self.fixed_task_count > 0:
            # Auto-calculate capacity: tasks ÷ servers (rounded up to ensure all tasks fit)
            import math
            auto_capacity = math.ceil(self.fixed_task_count / self.num_servers)
            self.max_server_capacity = auto_capacity
            print(f"Auto-calculated server capacity: {self.fixed_task_count} tasks ÷ {self.num_servers} servers = {auto_capacity} capacity per server")
    
    # Algorithm parameters
    max_matching_rounds: int = 20        # Maximum rounds for matching algorithm
    preference_randomization_prob: float = 0.2  # Probability of preference randomization
    
    # Energy model parameters
    idle_power: float = 0.01             # Idle power consumption (Watts)
    processing_power_factor: float = 1e-9 # Processing power factor (Watts per cycle)
    
    def validate(self) -> list[str]:
        """
        Validate configuration parameters
        
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        if self.num_users <= 0:
            errors.append("Number of users must be positive")
        
        if self.num_servers <= 0:
            errors.append("Number of servers must be positive")
        
        if self.max_server_capacity <= 0:
            errors.append("Maximum server capacity must be positive")
        
        if self.transmission_power <= 0:
            errors.append("Transmission power must be positive")
        
        if self.channel_bandwidth <= 0:
            errors.append("Channel bandwidth must be positive")
        
        if self.noise_power <= 0:
            errors.append("Noise power must be positive")
        
        if self.network_area_size <= 0:
            errors.append("Network area size must be positive")
        
        if self.min_task_data_size >= self.max_task_data_size:
            errors.append("Minimum task data size must be less than maximum")
        
        if self.min_task_priority >= self.max_task_priority:
            errors.append("Minimum task priority must be less than maximum")
        
        if self.server_frequency <= 0:
            errors.append("Server frequency must be positive")
        
        if not (0 <= self.preference_randomization_prob <= 1):
            errors.append("Preference randomization probability must be between 0 and 1")
        
        return errors
    
    def get_summary(self) -> str:
        """
        Get a formatted summary of the configuration
        
        Returns:
            Formatted configuration summary string
        """
        summary = []
        summary.append("=== System Configuration Summary ===")
        summary.append(f"Network Topology:")
        summary.append(f"  • Users: {self.num_users}")
        summary.append(f"  • Servers: {self.num_servers}")
        summary.append(f"  • Task Types: {self.num_task_types}")
        summary.append(f"  • Max Server Capacity: {self.max_server_capacity}")
        summary.append(f"  • Network Area: {self.network_area_size}×{self.network_area_size}m")
        
        summary.append(f"\nWireless Communication:")
        summary.append(f"  • Transmission Power: {self.transmission_power}W")
        summary.append(f"  • Channel Bandwidth: {self.channel_bandwidth/1e6:.1f}MHz")
        summary.append(f"  • Noise Power: {self.noise_power}W")
        summary.append(f"  • Path Loss Exponent: {self.path_loss_exponent}")
        
        summary.append(f"\nComputation:")
        summary.append(f"  • Server Frequency: {self.server_frequency/1e9:.1f}GHz")
        summary.append(f"  • Computation Density: {self.computation_density} cycles/bit")
        
        summary.append(f"\nTask Parameters:")
        summary.append(f"  • Data Size Range: {self.min_task_data_size/1e6:.1f}-{self.max_task_data_size/1e6:.1f}MB")
        summary.append(f"  • Priority Range: {self.min_task_priority}-{self.max_task_priority}")
        
        return "\n".join(summary)


class ConfigurationPresets:
    """Predefined configuration presets for different scenarios"""
    
    @staticmethod
    def small_scale() -> SystemConfiguration:
        """Small scale deployment (testing/development)"""
        return SystemConfiguration(
            num_users=3,
            num_servers=2,
            num_task_types=2,
            max_server_capacity=1,
            network_area_size=50.0
        )
    
    @staticmethod
    def medium_scale() -> SystemConfiguration:
        """Medium scale deployment (typical scenario)"""
        return SystemConfiguration(
            num_users=5,
            num_servers=4,
            num_task_types=3,
            max_server_capacity=2,
            network_area_size=100.0
        )
    
    @staticmethod
    def large_scale() -> SystemConfiguration:
        """Large scale deployment (stress testing)"""
        return SystemConfiguration(
            num_users=10,
            num_servers=6,
            num_task_types=4,
            max_server_capacity=3,
            network_area_size=200.0
        )
    
    @staticmethod
    def high_capacity() -> SystemConfiguration:
        """High capacity servers scenario"""
        return SystemConfiguration(
            num_users=8,
            num_servers=4,
            num_task_types=3,
            max_server_capacity=4,
            network_area_size=150.0,
            server_frequency=4e9  # Higher server frequency
        )
    
    @staticmethod
    def low_power() -> SystemConfiguration:
        """Low power scenario (IoT focused)"""
        return SystemConfiguration(
            num_users=6,
            num_servers=3,
            num_task_types=2,
            max_server_capacity=2,
            network_area_size=80.0,
            transmission_power=0.05,  # Lower transmission power
            min_task_data_size=5e5,   # Smaller tasks
            max_task_data_size=2e6
        )


def main():
    """Example usage of configuration classes"""
    print("=== Configuration Examples ===\n")
    
    # Default configuration
    default_config = SystemConfiguration()
    print("Default Configuration:")
    print(default_config.get_summary())
    
    # Validate configuration
    errors = default_config.validate()
    if errors:
        print(f"\nValidation Errors: {errors}")
    else:
        print(f"\n✓ Configuration is valid")
    
    # Preset configurations
    print(f"\n{'='*50}")
    print("Available Presets:")
    print("="*50)
    
    presets = {
        "Small Scale": ConfigurationPresets.small_scale(),
        "Medium Scale": ConfigurationPresets.medium_scale(),
        "Large Scale": ConfigurationPresets.large_scale(),
        "High Capacity": ConfigurationPresets.high_capacity(),
        "Low Power": ConfigurationPresets.low_power()
    }
    
    for name, config in presets.items():
        print(f"\n{name}:")
        print(f"  Users: {config.num_users}, Servers: {config.num_servers}")
        print(f"  Max Capacity: {config.max_server_capacity}")
        print(f"  Network Area: {config.network_area_size}m")
        print(f"  Server Freq: {config.server_frequency/1e9:.1f}GHz")


if __name__ == "__main__":
    main()