#!/bin/bash

# ── Configuration ────────────────────────────────────────────────
SCRIPTS_DIR="/c/users/fernando/Nextcloud/PostDoc/IPF/simulation/data/Ideal-chains"          # folder with your two .py files
ROOT_DIR="/c/users/fernando/Nextcloud/PostDoc/IPF/simulation/data/raw-data/f3"                # top-level folder to search under
PYTHON="python3"                             # or "python", or a venv path
# ────────────────────────────────────────────────────────────────

# Find every LastConfig.bfm, anywhere under ROOT_DIR
find "$ROOT_DIR" -name "LastConfig.bfm" | sort | while read -r bfm_file; do

    bfm_dir="$(dirname "$bfm_file")"
    echo "──────────────────────────────────────────"
    echo "Processing: $bfm_file"

    # Step into the folder so hardcoded filenames resolve correctly
    cd "$bfm_dir" || { echo "ERROR: cannot cd into $bfm_dir, skipping."; continue; }

    # Check if output folder exists
    #if [ -d "output" ]; then
    #    echo "  INFO: output folder already exists in $bfm_dir, skipping."
    #    continue
    #fi

    # Step 1 — prepare input
    echo "  Running extractPosConn.py ..."
    "$PYTHON" -u "$SCRIPTS_DIR/extractPosConn.py"
    if [ $? -ne 0 ]; then
        echo "  ERROR: extractPosConn.py failed in $bfm_dir, skipping minimization."
        continue
    fi

    # Step 2 — run minimization
    echo "  Running minimization-energybreak.py ..."
    "$PYTHON" -u "$SCRIPTS_DIR/minimization-energybreak.py"
    if [ $? -ne 0 ]; then
        echo "  ERROR: minimization-energybreak.py failed in $bfm_dir"
    fi

done

echo "══════════════════════════════════════════"
echo "All done."