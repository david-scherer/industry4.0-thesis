import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

def analyze_logs(sensor_log_files, scheduler_log_file):
    """
    Analyzes sensor and scheduler logs to calculate statistical metrics and create plots.

    Args:
        sensor_log_files: A list of sensor log file names.
        scheduler_log_file: The name of the scheduler log file.
    """

    # --- Sensor Log Analysis ---
    sensor_data = []
    for sensor_log in sensor_log_files:
        with open(f'/home/davestar/master-thesis/master-thesis/experiment-end-to-end/logs/{sensor_log}', 'r') as f:
            for line in f:
                parts = line.split(',')
                arrival_time_str = parts[1].split(': ')[1].strip()
                end_time_str = parts[2].split(': ')[1].strip()

                arrival_time = datetime.strptime(arrival_time_str, '%Y-%m-%d %H:%M:%S.%f')
                end_time = datetime.strptime(end_time_str, '%Y-%m-%d %H:%M:%S.%f')
                duration = (end_time - arrival_time).total_seconds() * 1000  # Duration in milliseconds
                sensor_data.append({'sensor': sensor_log, 'arrival_time': arrival_time, 'duration': duration})

    sensor_df = pd.DataFrame(sensor_data)

    # --- Scheduler Log Analysis ---
    scheduler_data = []
    with open(f'/home/davestar/master-thesis/master-thesis/experiment-end-to-end/logs/{scheduler_log_file}', 'r') as f:
        for line in f:
            parts = line.split(',')
            arrival_time_str = " ".join(parts[0].split(' ')[1:]).strip()
            scheduling_time_str = " ".join(parts[1].split(' ')[2:]).strip()
            duration_call_str = parts[2].split(' ')[3].strip()

            arrival_time = datetime.strptime(arrival_time_str, '%Y-%m-%d %H:%M:%S.%f')
            scheduling_time = datetime.strptime(scheduling_time_str, '%Y-%m-%d %H:%M:%S.%f')
            duration_call = float(duration_call_str) * 1000  # Duration in milliseconds

            scheduler_data.append({
                'arrival_time': arrival_time,
                'scheduling_time': scheduling_time,
                'duration_call': duration_call
            })

    scheduler_df = pd.DataFrame(scheduler_data)

    # Sort the `arrival_time` column in both DataFrames
    sensor_df = sensor_df.sort_values('arrival_time')
    scheduler_df = scheduler_df.sort_values('arrival_time')

    # --- Combine and Calculate Metrics ---
    # (Assuming arrival times can be used to align sensor and scheduler logs)
    merged_df = pd.merge_asof(sensor_df, scheduler_df, on='arrival_time', direction='nearest')
    merged_df['total_duration'] = merged_df['duration'] + merged_df['duration_call']

    # --- Statistical Metrics ---
    print("Sensor Statistics:")
    print(merged_df.groupby('sensor')['duration'].describe())

    print("\nScheduler Statistics:")
    print(merged_df['duration_call'].describe())

    print("\nTotal Single-trip Statistics:")
    print(merged_df['total_duration'].describe())

    # --- Plotting ---
    plt.figure(figsize=(12, 8))

    # Durations over Time
    plt.figure(figsize=(10, 6))
    for sensor in sensor_df['sensor'].unique():
        sensor_data = merged_df[merged_df['sensor'] == sensor]
        if "i=2" in sensor:
            label = "casting speed"
        elif "i=4" in sensor:
            label = "water temperature"
        elif "i=5" in sensor:
            label = "water flow"
        elif "i=7" in sensor:
            label = "metal temperature"
        else:
            label = "mois"
        plt.plot(sensor_data['arrival_time'], sensor_data['duration'], label=label)
    plt.xlabel('Time')
    plt.ylabel('Duration (ms)')
    plt.title('Sensor Durations Over Time')
    plt.legend()
    plt.tight_layout()
    plt.savefig("sensor_durations_over_time.png")

    # Distribution of Durations
    plt.figure(figsize=(10, 6))
    plt.hist(merged_df['duration'], alpha=0.5, label='Sensor')
    plt.hist(merged_df['duration_call'], alpha=0.5, label='Scheduler')
    plt.hist(merged_df['total_duration'], alpha=0.5, label='Total')
    plt.xlabel('Duration (ms)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Durations')
    plt.legend()
    plt.tight_layout()
    plt.savefig("distribution_of_durations.png")

    # Scatter Plot - Single-trip vs. Sensor Duration
    plt.figure(figsize=(10, 6))
    plt.scatter(merged_df['duration'], merged_df['total_duration'], alpha=0.5)
    plt.xlabel('Sensor Duration (ms)')
    plt.ylabel('Total Duration (ms)')
    plt.title('Single-trip vs. Sensor Duration')
    plt.tight_layout()
    plt.savefig("single_trip_vs_sensor_duration.png")

    # Scatter Plot - Single-trip vs. Scheduler Duration
    plt.figure(figsize=(10, 6))
    plt.scatter(merged_df['duration_call'], merged_df['total_duration'], alpha=0.5)
    plt.xlabel('Scheduler Duration (ms)')
    plt.ylabel('Total Duration (ms)')
    plt.title('Single-trip vs. Scheduler Duration')
    plt.tight_layout()
    plt.savefig("single_trip_vs_scheduler_duration.png")
# --- Main Execution ---
sensor_files = ["logs-ns=2;i=2.log", "logs-ns=2;i=4.log", "logs-ns=2;i=5.log", "logs-ns=2;i=7.log"]
scheduler_file = "scheduler.log"

analyze_logs(sensor_files, scheduler_file)