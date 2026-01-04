#!/bin/bash
# Installation script to copy optimizations to the ffa-q2fp8-threshold project

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}========================================================================"
echo "Q2FP8 Optimizations - Installation Script"
echo -e "========================================================================${NC}"
echo ""

# Get the source directory (where this script is)
SOURCE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Check if we're in the right place
if [ ! -d "${SOURCE_DIR}/kernels" ] || [ ! -f "${SOURCE_DIR}/README.md" ]; then
    echo -e "${RED}ERROR: This script must be run from the q2fp8_optimizations directory${NC}"
    echo "Current directory: ${SOURCE_DIR}"
    exit 1
fi

# Ask for target directory
echo "This script will copy the optimization files to your ffa-q2fp8-threshold project."
echo ""
echo -e "${YELLOW}Please enter the path to your ffa-q2fp8-threshold project:${NC}"
read -p "Path: " TARGET_DIR

# Expand tilde
TARGET_DIR="${TARGET_DIR/#\~/$HOME}"

# Validate target directory
if [ ! -d "${TARGET_DIR}" ]; then
    echo -e "${RED}ERROR: Directory does not exist: ${TARGET_DIR}${NC}"
    exit 1
fi

if [ ! -d "${TARGET_DIR}/attn_kernel" ]; then
    echo -e "${YELLOW}WARNING: attn_kernel directory not found in ${TARGET_DIR}${NC}"
    echo "Are you sure this is the correct ffa-q2fp8-threshold project directory?"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Installation cancelled."
        exit 1
    fi
fi

echo ""
echo -e "${GREEN}Installing to: ${TARGET_DIR}${NC}"
echo ""

# Create directory structure
echo "Creating directory structure..."
mkdir -p "${TARGET_DIR}/optimizations_v2/kernels"
mkdir -p "${TARGET_DIR}/optimizations_v2/benchmarks"
mkdir -p "${TARGET_DIR}/optimizations_v2/results"

# Copy files
echo "Copying kernel files..."
cp -v "${SOURCE_DIR}/kernels/"*.py "${TARGET_DIR}/optimizations_v2/kernels/"

echo "Copying benchmark files..."
cp -v "${SOURCE_DIR}/benchmarks/"*.py "${TARGET_DIR}/optimizations_v2/benchmarks/"
chmod +x "${TARGET_DIR}/optimizations_v2/benchmarks/benchmark_all_opts.py"

echo "Copying scripts..."
cp -v "${SOURCE_DIR}/run_all_benchmarks.sh" "${TARGET_DIR}/optimizations_v2/"
chmod +x "${TARGET_DIR}/optimizations_v2/run_all_benchmarks.sh"

cp -v "${SOURCE_DIR}/quick_test.py" "${TARGET_DIR}/optimizations_v2/"
chmod +x "${TARGET_DIR}/optimizations_v2/quick_test.py"

echo "Copying documentation..."
cp -v "${SOURCE_DIR}/README.md" "${TARGET_DIR}/optimizations_v2/"

echo ""
echo -e "${GREEN}✓ Installation complete!${NC}"
echo ""
echo -e "${BLUE}========================================================================"
echo "Next Steps"
echo -e "========================================================================${NC}"
echo ""
echo "1. Navigate to the installation directory:"
echo -e "   ${YELLOW}cd ${TARGET_DIR}/optimizations_v2${NC}"
echo ""
echo "2. Run quick test to verify kernels work:"
echo -e "   ${YELLOW}python3 quick_test.py${NC}"
echo ""
echo "3. If tests pass, run full benchmarks:"
echo -e "   ${YELLOW}./run_all_benchmarks.sh${NC}"
echo ""
echo "4. Review results in the results_<timestamp>/ directory"
echo ""
echo "5. Copy the directory to H100 and run again for comparison"
echo ""
echo "For more details, see: ${TARGET_DIR}/optimizations_v2/README.md"
echo ""

exit 0
