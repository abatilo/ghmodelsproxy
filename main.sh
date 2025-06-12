#!/bin/bash

for i in {1..15}; do
    echo "=== Run $i ==="
    go run main.go -headers -model "openai/o3" "respond as quickly as possible"
    echo
    
    # Sleep for 30 seconds between runs (except after the last run)
    if [ $i -lt 15 ]; then
        echo "Sleeping for 30 seconds..."
        sleep 30
    fi
done