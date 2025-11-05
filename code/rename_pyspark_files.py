#!/usr/bin/env python3
import os
import json
import glob

results_dir = "../data/results/pyspark/"

# Get all pyspark files that don't have aws or gcp in the name
files = glob.glob(os.path.join(results_dir, "pyspark_*.json"))
files_to_rename = [f for f in files if "_aws_" not in f and "_gcp_" not in f]

print(f"Found {len(files_to_rename)} files to rename")

for filepath in files_to_rename:
    # Read the file to get the provider
    with open(filepath, 'r') as f:
        data = json.load(f)
        provider = data.get('provider', 'unknown')

    # Generate new filename
    filename = os.path.basename(filepath)
    # Insert provider after "pyspark_"
    new_filename = filename.replace("pyspark_", f"pyspark_{provider}_")
    new_filepath = os.path.join(results_dir, new_filename)

    # Rename the file
    print(f"Renaming: {filename} -> {new_filename}")
    os.rename(filepath, new_filepath)

print("Done!")
