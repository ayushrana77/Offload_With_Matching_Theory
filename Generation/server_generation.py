"""
Server/Node Generation Module
Handles generation of edge, fog, and cloud servers
"""

import random
import numpy as np
from typing import Dict, List, Tuple


class ServerGenerator:
    """
    Generator for edge, fog, and cloud servers
    Supports both single-level and multi-level hierarchical architectures
    """
    
    def __init__(self, config):
        """
        Initialize server generator
        
        Args:
            config: SystemConfiguration instance
        """
        self.config = config
    
    def generate_servers(self) -> List[Dict]:
        """
        Generate servers based on configuration
        Automatically detects single-level or multi-level architecture
        
        Returns:
            List of server dictionaries with computational capabilities
        """
        if hasattr(self.config, 'use_multilevel') and self.config.use_multilevel:
            servers = self._generate_multilevel_servers()
            print(f"✅ Generated {len(servers)} servers (multi-level architecture)")
        else:
            servers = self._generate_single_level_servers()
            print(f"✅ Generated {len(servers)} servers (single-level architecture)")
        
        return servers
    
    def _generate_single_level_servers(self) -> List[Dict]:
        """
        Generate servers for single-level architecture
        All servers have similar capabilities
        
        Returns:
            List of server dictionaries
        """
        servers = []
        
        # Get server capacity range from config
        min_capacity = getattr(self.config, 'min_server_capacity', 1e9)
        max_capacity = getattr(self.config, 'max_server_capacity', 5e9)
        
        for i in range(self.config.num_servers):
            server = {
                'id': f'S{i+1}',
                'level': 1,
                'type': 'fog',
                'position': self._generate_random_position(),
                'computational_capability': random.uniform(min_capacity, max_capacity),
                'available_resources': random.uniform(0.5, 1.0),  # 50-100% available
                'processing_cost': random.uniform(0.01, 0.05),  # Cost per cycle
                'communication_delay_base': 0.001,  # 1ms base delay
                'bandwidth': random.uniform(10e6, 100e6),  # 10-100 Mbps
                'energy_efficiency': random.uniform(0.5, 1.0)  # Energy efficiency factor
            }
            servers.append(server)
        
        return servers
    
    def _generate_multilevel_servers(self) -> List[Dict]:
        """
        Generate servers for multi-level architecture
        Creates edge, regional fog, and cloud servers
        
        Returns:
            List of server dictionaries with level-specific properties
        """
        servers = []
        
        # Level 1: Edge Fog Servers (Local, low latency, limited capacity)
        edge_servers = self._generate_edge_servers()
        servers.extend(edge_servers)
        
        # Level 2: Regional Fog Servers (Intermediate, medium latency, higher capacity)
        regional_servers = self._generate_regional_servers()
        servers.extend(regional_servers)
        
        # Level 3: Cloud Servers (Centralized, high latency, massive capacity)
        cloud_servers = self._generate_cloud_servers()
        servers.extend(cloud_servers)
        
        return servers
    
    def _generate_edge_servers(self) -> List[Dict]:
        """
        Generate Level 1: Edge fog servers
        Characteristics: Distributed, low latency, limited resources
        
        Returns:
            List of edge server dictionaries
        """
        num_edge = getattr(self.config, 'edge_fog_servers', 
                          max(3, self.config.num_servers // 2))
        
        edge_servers = []
        
        for i in range(num_edge):
            # Edge servers are distributed across the network area
            server = {
                'id': f'E{i+1}',
                'level': 1,
                'type': 'edge_fog',
                'name': f'Edge Server {i+1}',
                'position': self._generate_random_position(),
                'computational_capability': random.uniform(2e9, 5e9),  # 2-5 GHz
                'available_resources': random.uniform(0.6, 0.9),
                'processing_cost': random.uniform(0.02, 0.04),  # Higher cost (limited resources)
                'communication_delay_base': 0.001,  # 1ms - very low latency
                'bandwidth': random.uniform(50e6, 200e6),  # 50-200 Mbps
                'energy_efficiency': random.uniform(0.6, 0.9),
                'coverage_radius': random.uniform(50, 100),  # meters
                'reliability': random.uniform(0.9, 0.95)
            }
            edge_servers.append(server)
        
        return edge_servers
    
    def _generate_regional_servers(self) -> List[Dict]:
        """
        Generate Level 2: Regional fog servers
        Characteristics: Medium latency, higher capacity than edge
        
        Returns:
            List of regional server dictionaries
        """
        num_regional = getattr(self.config, 'regional_fog_servers',
                              max(2, self.config.num_servers // 3))
        
        regional_servers = []
        
        # Regional servers are placed strategically (e.g., in grid pattern)
        grid_size = int(np.ceil(np.sqrt(num_regional)))
        step_size = self.config.network_area_size / (grid_size + 1)
        
        for i in range(num_regional):
            row = i // grid_size
            col = i % grid_size
            
            # Grid positioning with some randomness
            base_x = (col + 1) * step_size
            base_y = (row + 1) * step_size
            
            server = {
                'id': f'R{i+1}',
                'level': 2,
                'type': 'regional_fog',
                'name': f'Regional Server {i+1}',
                'position': (
                    base_x + random.uniform(-step_size/3, step_size/3),
                    base_y + random.uniform(-step_size/3, step_size/3)
                ),
                'computational_capability': random.uniform(5e9, 10e9),  # 5-10 GHz
                'available_resources': random.uniform(0.7, 0.95),
                'processing_cost': random.uniform(0.015, 0.03),  # Medium cost
                'communication_delay_base': 0.005,  # 5ms delay
                'bandwidth': random.uniform(100e6, 500e6),  # 100-500 Mbps
                'energy_efficiency': random.uniform(0.7, 0.95),
                'coverage_radius': random.uniform(100, 200),  # meters
                'reliability': random.uniform(0.95, 0.98)
            }
            regional_servers.append(server)
        
        return regional_servers
    
    def _generate_cloud_servers(self) -> List[Dict]:
        """
        Generate Level 3: Cloud servers
        Characteristics: Centralized, high latency, virtually unlimited resources
        
        Returns:
            List of cloud server dictionaries
        """
        num_cloud = getattr(self.config, 'cloud_servers',
                           max(1, self.config.num_servers // 5))
        
        cloud_servers = []
        
        # Cloud servers are typically at the center or fixed locations
        center = self.config.network_area_size / 2
        
        for i in range(num_cloud):
            # Position near center with some offset
            offset = i * 20  # Slight offset for multiple cloud servers
            
            server = {
                'id': f'C{i+1}',
                'level': 3,
                'type': 'cloud',
                'name': f'Cloud Server {i+1}',
                'position': (center + offset, center + offset),
                'computational_capability': random.uniform(10e9, 20e9),  # 10-20 GHz
                'available_resources': random.uniform(0.8, 1.0),  # Always high availability
                'processing_cost': random.uniform(0.01, 0.02),  # Lower cost (economies of scale)
                'communication_delay_base': 0.01,  # 10ms delay
                'bandwidth': random.uniform(500e6, 2e9),  # 500 Mbps - 2 Gbps
                'energy_efficiency': random.uniform(0.8, 1.0),
                'coverage_radius': float('inf'),  # Global coverage
                'reliability': random.uniform(0.98, 0.999)  # Very high reliability
            }
            cloud_servers.append(server)
        
        return cloud_servers
    
    def _generate_random_position(self) -> Tuple[float, float]:
        """Generate random position within network area"""
        return (
            random.uniform(0, self.config.network_area_size),
            random.uniform(0, self.config.network_area_size)
        )
    
    def get_server_summary(self, servers: List[Dict]) -> Dict:
        """
        Get summary statistics for generated servers
        
        Args:
            servers: List of server dictionaries
            
        Returns:
            Dictionary with summary statistics
        """
        by_level = {}
        by_type = {}
        
        for server in servers:
            level = server['level']
            server_type = server['type']
            
            by_level[level] = by_level.get(level, 0) + 1
            by_type[server_type] = by_type.get(server_type, 0) + 1
        
        return {
            'total_servers': len(servers),
            'by_level': by_level,
            'by_type': by_type,
            'avg_capability': np.mean([s['computational_capability'] for s in servers]),
            'avg_cost': np.mean([s['processing_cost'] for s in servers]),
            'total_capacity': sum(s['computational_capability'] for s in servers)
        }
    
    def print_server_hierarchy(self, servers: List[Dict]):
        """
        Print server hierarchy information
        
        Args:
            servers: List of server dictionaries
        """
        print("\n🏗️  SERVER HIERARCHY:")
        
        # Group by level
        by_level = {}
        for server in servers:
            level = server['level']
            if level not in by_level:
                by_level[level] = []
            by_level[level].append(server)
        
        # Print each level
        level_names = {1: "EDGE FOG", 2: "REGIONAL FOG", 3: "CLOUD"}
        
        for level in sorted(by_level.keys()):
            level_servers = by_level[level]
            level_name = level_names.get(level, f"LEVEL {level}")
            
            print(f"\n  📊 {level_name} (Level {level}):")
            print(f"     Servers: {len(level_servers)}")
            print(f"     Avg Capability: {np.mean([s['computational_capability'] for s in level_servers])/1e9:.2f} GHz")
            print(f"     Avg Cost: {np.mean([s['processing_cost'] for s in level_servers]):.4f}")
            print(f"     Avg Delay: {np.mean([s['communication_delay_base'] for s in level_servers])*1000:.1f} ms")


def main():
    """Test server generation module"""
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    
    from config import SystemConfiguration
    
    print("="*70)
    print("Testing Server Generation Module")
    print("="*70)
    
    # Test single-level architecture
    print("\n1️⃣  Single-Level Architecture:")
    config_single = SystemConfiguration(
        num_servers=5,
        random_seed=42
    )
    
    random.seed(42)
    np.random.seed(42)
    
    server_gen = ServerGenerator(config_single)
    servers_single = server_gen.generate_servers()
    summary_single = server_gen.get_server_summary(servers_single)
    
    print(f"  Total servers: {summary_single['total_servers']}")
    print(f"  Types: {summary_single['by_type']}")
    
    # Test multi-level architecture
    print("\n2️⃣  Multi-Level Architecture:")
    config_multi = SystemConfiguration(
        num_servers=10,
        use_multilevel=True,
        edge_fog_servers=4,
        regional_fog_servers=2,
        cloud_servers=1,
        random_seed=42
    )
    
    random.seed(42)
    np.random.seed(42)
    
    server_gen_multi = ServerGenerator(config_multi)
    servers_multi = server_gen_multi.generate_servers()
    summary_multi = server_gen_multi.get_server_summary(servers_multi)
    
    print(f"  Total servers: {summary_multi['total_servers']}")
    print(f"  By level: {summary_multi['by_level']}")
    print(f"  By type: {summary_multi['by_type']}")
    
    server_gen_multi.print_server_hierarchy(servers_multi)
    
    print("\n✅ Server Generation Module Test Complete")


if __name__ == "__main__":
    main()
