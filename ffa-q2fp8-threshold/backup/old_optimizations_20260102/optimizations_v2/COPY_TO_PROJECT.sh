#!/bin/bash
# Script to copy optimizations to project directory
# Run with sudo if needed

TARGET="/inspire/qb-ilm/project/exploration-topic/liuzhigeng-253108120105/projects/ffa/ffa-speedtest/ffa-q2fp8-threshold/optimizations_v2"

echo "This will copy optimizations to:"
echo "  $TARGET"
echo ""
echo "You may need to run with sudo:"
echo "  sudo bash COPY_TO_PROJECT.sh"
echo ""

# Create directory
mkdir -p "$TARGET"

# Copy files
cp -rv ~/q2fp8_test/* "$TARGET/"

echo ""
echo "Done! Files copied to $TARGET"
