from datetime import datetime
import numpy as np
import sys

# Function to parse the log file and calculate energy consumption and total time
def calculate_energy_consumption_and_total_time(log_file):
    gpu_energy = {}
    gpu_last_timestamp = {}
    first_timestamp = None
    last_timestamp = None

    with open(log_file, 'r') as file:
        for line in file:
            line=line.rstrip()
            # Skip comments
            if line.startswith("time"):
                continue
            
            # Split the line into columns
            columns = line.split(",")
            #time,gpu_id,memory_used_MB,utilization_percent,temperature_C,power_draw_W

            dt=columns[0]
            date_str=dt.split()[0]
            time_str=dt.split()[1]
            gpu_idx = int(columns[1])
            power = float(columns[5])  # Power in watts
    
            # Combine date and time into a single datetime object
            current_timestamp = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")

            # Track the first and last timestamps for the entire log
            if first_timestamp is None:
                first_timestamp = current_timestamp
            last_timestamp = current_timestamp

            # Check if this is the first entry for this GPU
            if gpu_idx not in gpu_last_timestamp:
                gpu_last_timestamp[gpu_idx] = current_timestamp
                gpu_energy[gpu_idx] = 0.0  # Initialize energy for this GPU
                continue

            # Calculate time difference in hours for this GPU
            time_diff = (current_timestamp - gpu_last_timestamp[gpu_idx]).total_seconds() / 3600.0

            # Calculate energy consumption for this interval
            energy_consumption = power * time_diff  # Energy in watt-hours (Wh)

            # Accumulate the energy consumption per GPU
            gpu_energy[gpu_idx] += energy_consumption

            # Update last timestamp for this GPU
            gpu_last_timestamp[gpu_idx] = current_timestamp

    # Convert Wh to kWh
    for gpu_idx in gpu_energy:
        gpu_energy[gpu_idx] /= 1000.0  # Convert to kWh

    # Calculate total consumption across all GPUs
    total_consumption = np.sum(list(gpu_energy.values()))

    # Calculate total time duration (from the first entry of any GPU to the last entry of any GPU)
    if first_timestamp is not None and last_timestamp is not None:
        total_time_seconds = (last_timestamp - first_timestamp).total_seconds()
        total_time_hours = total_time_seconds / 3600.0

        # Convert total time to hours and minutes
        total_hours = int(total_time_hours)
        total_minutes = int((total_time_hours - total_hours) * 60)
    else:
        total_time_seconds = 0
        total_time_hours = 0
        total_hours = 0
        total_minutes = 0

    return gpu_energy, total_consumption, total_time_seconds, total_hours, total_minutes

# File path to your log file

log_file = sys.argv[1]
# Calculate energy consumption and total time
gpu_energy, total_consumption, total_time_seconds, total_hours, total_minutes = calculate_energy_consumption_and_total_time(log_file)

# Output the results
print("Energy consumption per GPU (kWh):")
for gpu_idx, energy in gpu_energy.items():
    print(f"GPU {gpu_idx}: {energy:.3f} kWh")

print(f"Total energy consumption (kWh): {total_consumption:.3f} kWh")
print(f"Total time: {total_hours} hours and {total_minutes} minutes")
