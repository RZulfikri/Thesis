#!/bin/bash
# Get the directory where the script is located
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Check if venv exists
if [ ! -d "$DIR/venv" ]; then
    echo "Virtual environment not found. Please run setup first."
    exit 1
fi

# Run the python script using the venv interpreter
"$DIR/venv/bin/python3" "$DIR/run.py" "$@"
