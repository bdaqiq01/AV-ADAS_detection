import argparse
import math
import statistics
from tabulate import tabulate

def extract_fps(logfile):
    fps_values = []
    with open(logfile, "r") as f:
        for line in f:
            split = line.split('|')[1]
            value = split.split(': ')[1]
            fps_values.append(float(value))
    return fps_values

def main():
    parser = argparse.ArgumentParser(description="Calculate FPS statistics from log file")
    parser.add_argument("logfile", help="Path to the log file")
    args = parser.parse_args()

    fps = extract_fps(args.logfile)

    n = len(fps)
    z = 1.96
    std_dev = statistics.stdev(fps)
    margin_error_95 = z * (std_dev / math.sqrt(n))

    data = [
        ["No. of Entries", n],
        ["Min FPS", min(fps)],
        ["Max FPS", max(fps)],
        ["Mean FPS", statistics.mean(fps)],
        ["Median FPS", statistics.median(fps)],
        ["Std Dev", std_dev],
        ["95% Margin of Error", margin_error_95],
    ]

    headers = ["Metric", "Value"]

    print(tabulate(data, headers=headers, tablefmt="simple_outline", colalign=("left", "center")))


if __name__ == "__main__":
    main()

