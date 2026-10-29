#!/usr/bin/env python3
"""
This script measures GPU memory usage using the `nvidia-smi` command-line tool. It extracts the used and total memory
every second for each GPU and saves the results to a CSV file. The script can be run in the background and will
continue to log GPU memory usage until it is stopped. After stopping, the script will also plot the memory usage over
time.

Usage:
    python measure_gpu_usage.py --gpu-ids <GPU_IDS> --output <OUTPUT_CSV_FILE>
"""

import json
import logging
import re
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt

try:
    import rich_click as click
except ImportError:
    import click

logger = logging.getLogger(__name__)
now = datetime.now
strptime = datetime.strptime


def main(gpu_ids: tuple[int] | None, output: Path, frequency: float):
    """
    Measures GPU memory usage and saves the results to a CSV file.

    Parameters
    ----------
    gpu_ids : tuple[int] | None
        Tuple of GPU IDs to monitor. If None, all GPUs will be monitored.
    output : Path
        Path to the output CSV file.
    """
    # Ensure the output directory exists
    frequency = max(frequency, 0.1)  # Ensure frequency is at least 0.1 seconds
    output.parent.mkdir(parents=True, exist_ok=True)

    # Prepare the command to run nvidia-smi
    command = ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"]
    if gpu_ids:
        command.append(f"--id={','.join(map(str, gpu_ids))}")

    # Open the output file for writing
    with open(output, "w") as f:
        f.write("timestamp,gpu_id,memory_used,memory_total\n")  # Write header

        try:
            start_time = now()
            i = 0
            while True:
                # Run the nvidia-smi command and capture the output
                timestamp = now().strftime("%Y-%m-%d %H:%M:%S")
                result = subprocess.run(command, capture_output=True, text=True, check=True)

                # Parse the output and write to the CSV file
                for gpu_id, line in enumerate(result.stdout.strip().splitlines()):
                    memory_used, memory_total = map(int, re.findall(r"\d+", line))
                    f.write(f"{timestamp},{gpu_id},{memory_used},{memory_total}\n")
                    f.flush()  # Ensure data is written to the file immediately

                i += 1
                time.sleep(
                    (start_time + timedelta(seconds=frequency) * i - now()).total_seconds()
                )  # Sleep until the next interval

        except (KeyboardInterrupt, SystemExit):
            print()
            logger.info("Measurement stopped by user.")
            logger.info(f"Results saved to '{output}'")

            # Load the data from the CSV file
            data = {}
            with open(output, "r") as f:
                for line in f.readlines()[1:]:  # Skip header
                    timestamp, gpu_id, memory_used, memory_total = line.strip().split(",")
                    if gpu_id not in data:
                        data[gpu_id] = {"timestamps": [], "memory_used": []}
                    data[gpu_id]["timestamps"].append(timestamp)
                    data[gpu_id]["memory_used"].append(int(memory_used))

            if len(data) == 0:
                return

            # Print statistics about the GPU memory usage
            logger.info("GPU Memory Usage Statistics:")
            stats = {}
            for gpu_id, gpu_data in data.items():
                max_memory = max(gpu_data["memory_used"])
                min_memory = min(gpu_data["memory_used"])
                avg_memory = sum(gpu_data["memory_used"]) / len(gpu_data["memory_used"])
                stats[gpu_id] = {"max": max_memory, "min": min_memory, "avg": avg_memory}
                logger.info(
                    f"""  - GPU {gpu_id}: Max Memory Used: {max_memory} MiB
           Min Memory Used: {min_memory} MiB
           Avg Memory Used: {avg_memory:.2f} MiB"""
                )

            # Save the statistics to a json file
            with open(output.with_suffix(".json"), "w") as f:
                json.dump(stats, f, indent=4)

            # Plot the GPU memory usage over time
            plt.figure(figsize=(10, 6))
            seconds = []
            for gpu_id, gpu_data in data.items():
                timestamps = [strptime(t, "%Y-%m-%d %H:%M:%S") for t in gpu_data["timestamps"]]
                seconds = [(t - timestamps[0]).total_seconds() for t in timestamps]
                plt.plot(seconds, gpu_data["memory_used"], label=f"GPU {gpu_id}")
            plt.xlabel("Time (s)")
            plt.ylabel("Memory Used (MiB)")
            plt.title("GPU Memory Usage Over Time")
            plt.legend()
            num_ticks = 10
            tick_positions = [int(i * (len(seconds) - 1) / (num_ticks - 1)) for i in range(num_ticks)]
            plt.xticks([seconds[i] for i in tick_positions], [f"{seconds[i]:.0f}" for i in tick_positions])
            plt.tight_layout()
            logger.info(f"Saving plot to '{output.with_suffix('.png')}'")
            plt.savefig(output.with_suffix(".png"))  # Save the plot as a PNG file
            plt.close()


@click.command()
@click.option(
    "--gpu-ids",
    multiple=True,
    default=None,
    help="GPU IDs to monitor (e.g., 0,1,2). If not specified, all GPUs will be monitored.",
)
@click.option(
    "--output",
    default="gpu_usage.csv",
    help="Output CSV file path",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
)
@click.option(
    "--frequency",
    default=1.0,
    help="Measurement frequency in seconds",
    type=float,
)
def cli(gpu_ids: tuple[int], output: Path, frequency: float):
    """
    Measures GPU memory usage and saves the results to a CSV file.
    """
    main(gpu_ids, output, frequency)


if __name__ == "__main__":
    try:
        from rich.logging import RichHandler

        FORMAT = "%(message)s"
        logging.basicConfig(
            level="INFO",
            format=FORMAT,
            handlers=[RichHandler(rich_tracebacks=True, tracebacks_show_locals=True)],
        )
    except ImportError:
        FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        logging.basicConfig(level=logging.INFO, format=FORMAT, datefmt="%H:%M:%S")

    cli()
