#!/bin/bash

# OpenClaw Workspace Symlink Restorer
# Usage: Run this script inside ~/.openclaw/workspace/ on the remote machine
# or ensure CS-Notes is in the current directory.

WORKSPACE_DIR="$HOME/.openclaw/workspace"
SOURCE_DIR="CS-Notes/.openclaw-memory"

# List of files/directories that should be symlinks
FILES_TO_LINK=(
    "AGENTS.md"
    "HEARTBEAT.md"
    "IDENTITY.md"
    "MEMORY.md"
    "SOUL.md"
    "TOOLS.md"
    "USER.md"
    "memory"
)

echo "Starting OpenClaw symlink restoration..."
echo "Workspace: $WORKSPACE_DIR"
echo "Source: $SOURCE_DIR"

cd "$WORKSPACE_DIR" || { echo "Error: Cannot change to workspace directory $WORKSPACE_DIR"; exit 1; }

if [ ! -d "CS-Notes" ]; then
    echo "Error: CS-Notes directory not found in $WORKSPACE_DIR."
    echo "Please ensure you have cloned the repo or are in the right directory."
    exit 1
fi

for item in "${FILES_TO_LINK[@]}"; do
    target="$SOURCE_DIR/$item"
    
    # Check if the source file actually exists in CS-Notes
    if [ ! -e "$target" ]; then
        echo "Warning: Source file $target does not exist. Skipping..."
        continue
    fi

    # Check if the destination exists
    if [ -e "$item" ]; then
        # Check if it is already a symlink
        if [ -L "$item" ]; then
            current_link=$(readlink "$item")
            if [ "$current_link" == "$target" ]; then
                echo "[OK] $item is already correctly linked."
                continue
            else
                echo "[FIX] $item is a symlink but points to '$current_link'. Relinking..."
                rm "$item"
            fi
        else
            echo "[FIX] $item is a regular file/directory. Deleting and linking..."
            # Back up just in case, or just delete as requested
            # mv "$item" "${item}.bak" 
            rm -rf "$item"
        fi
    else
        echo "[NEW] $item does not exist. Creating link..."
    fi

    # Create the symlink
    ln -s "$target" "$item"
    echo "     -> Linked $item to $target"
done

echo "Restoration complete."
ls -laH "${FILES_TO_LINK[@]}"
