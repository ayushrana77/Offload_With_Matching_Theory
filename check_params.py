
import sys
import os
sys.path.append(os.getcwd())
from Algorithm.algorithm_utilities import AlgorithmUtilities
from config import SystemConfiguration

config = SystemConfiguration(
    num_users=10,
    num_servers=5,
    fixed_task_count=1000,
    use_multilevel=True
)

print("Generating system...")
users = AlgorithmUtilities.generate_users(config)
servers = AlgorithmUtilities.generate_servers(config)
tasks = AlgorithmUtilities.generate_tasks(config, users, servers)

print("\n--- Diagnostic Data ---")
server_caps = [s['computational_capability'] for s in servers]
print(f"Server Capabilities (Hz): {[f'{x/1e9:.2f}G' for x in server_caps]}")
print(f"Avg Server Cap: {sum(server_caps)/len(server_caps)/1e9:.2f} GHz")

task_reqs = [t['computation_requirement'] for t in tasks]
print(f"Task Requirements (Cycles): {[f'{x/1e6:.1f}M' for x in task_reqs[:10]]} ...")
print(f"Avg Task Req: {sum(task_reqs)/len(task_reqs)/1e6:.2f} M-cycles")

avg_proc_time = (sum(task_reqs)/len(task_reqs)) / (sum(server_caps)/len(server_caps))
print(f"Average Theoretical Processing Time: {avg_proc_time*1000:.4f} ms")

total_work = sum(task_reqs)
total_capacity = sum(server_caps)
print(f"Total Work Load: {total_work/1e9:.2f} G-cycles")
print(f"Total System Capacity: {total_capacity/1e9:.2f} G-cycles/sec")
print(f"Min System Clearance Time (perfect parallel): {total_work/total_capacity:.4f} s")
print(f"Configuration Initial Server Capacity: {config.initial_server_capacity}")
