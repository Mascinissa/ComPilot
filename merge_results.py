#!/usr/bin/env python3
"""Merge per-worker JSON result files into a single consolidated results file.

Usage:
    python merge_results.py --results-dir <results_dir>

Finds all results_*_worker*.json files in the given directory, merges their
program entries into a single results JSON file, and reports any missing programs.
"""

import argparse
import json
from pathlib import Path


def merge_worker_results(results_dir):
    results_dir = Path(results_dir)
    worker_files = sorted(results_dir.glob('results_*_worker*.json'))

    if not worker_files:
        print(f"No worker result files found in {results_dir}")
        return

    print(f"Found {len(worker_files)} worker file(s)")

    # Use the first worker's configuration as the base
    with open(worker_files[0], 'r') as f:
        merged = json.load(f)

    # Merge programs from all other worker files
    for wf in worker_files[1:]:
        with open(wf, 'r') as f:
            worker_data = json.load(f)
        for prog_name, prog_data in worker_data['programs'].items():
            if prog_name in merged['programs']:
                print(f"  Warning: duplicate program '{prog_name}' found in {wf.name}, skipping")
            else:
                merged['programs'][prog_name] = prog_data

    # Sort programs by name for consistent ordering
    merged['programs'] = dict(sorted(merged['programs'].items()))

    # Derive output filename from worker filename (strip _worker{id} suffix)
    base_name = worker_files[0].stem
    # Remove _worker0 (or any _worker{N}) suffix
    import re
    output_name = re.sub(r'_worker\d+$', '', base_name)
    output_path = results_dir / f'{output_name}.json'

    with open(output_path, 'w') as f:
        json.dump(merged, f, indent=2)

    print(f"Merged {len(merged['programs'])} programs into {output_path}")
    return output_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Merge per-worker ComPilot result files')
    parser.add_argument('--results-dir', required=True, help='Directory containing worker result JSON files')
    args = parser.parse_args()

    merge_worker_results(args.results_dir)
