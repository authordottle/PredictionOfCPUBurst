import csv
import psutil
import os

processes = []

def get_processes_to_csv(csv_path):
    # Get all running processes
    processes = psutil.process_iter()

    # Create a CSV file and write the header row
    with open(csv_path, mode='w', newline='') as csv_file:
        fieldnames = ['pid', 'name', 'username', 'cpu_percent', 'memory_percent']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        # Write each process's data to the CSV file
        for process in processes:
            try:
                process_data = process.as_dict(attrs=['pid', 'name', 'username', 'cpu_percent', 'memory_percent'])
            except psutil.NoSuchProcess:
                # Process may have terminated since we got the process list
                continue

            writer.writerow(process_data)

get_processes_to_csv('data/mac_processes.csv')