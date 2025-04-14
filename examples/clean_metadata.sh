#!/bin/bash

# BE VERY CAREFUL WITH THIS SCRIPT

# Check if both directories were provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <directory1> <directory2>"
    echo "This script deletes subfolders from directory1 that don't exist in directory2"
    exit 1
fi

D1="$1"
D2="$2"

# Check if directories exist
if [ ! -d "$D1" ]; then
    echo "Error: Directory $D1 does not exist."
    exit 1
fi

if [ ! -d "$D2" ]; then
    echo "Error: Directory $D2 does not exist."
    exit 1
fi

# Process each subfolder in d1
for subfolder in "$D1"/*; do
    # Check if it's a directory
    if [ -d "$subfolder" ]; then
        # Get the basename of the subfolder
        base=$(basename "$subfolder")
        
        # Check if a matching subfolder exists in d2
        if [ ! -d "$D2/$base" ]; then
            echo "Deleting $subfolder (not found in $D2)"
            rm -rf "$subfolder"
        fi
    fi
done

echo "Finished removing non-matching subfolders from $D1"