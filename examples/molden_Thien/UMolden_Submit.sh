#!/bin/bash

# ==== CẤU HÌNH ====
CPUS=4
mem=8192

source ~/venvs/bin/activate

export BASIS='ccpvdz'

export MOLE='Si3'
export NCORE=0
export ACT_ORB='full'
export ARRANGE='triplet-triangle'



SCRIPT_FILE="UMolden.py"


JOB_NAME="UMolden_${MOLE}_${ARRANGE}"

# Lấy đường dẫn hiện tại
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Thư mục output
OUTPUT_DIR="$SCRIPT_DIR/output"
mkdir -p "$OUTPUT_DIR"

# File tạm để submit
TEMP_JOB="$SCRIPT_DIR/UMolden_temp.sh"

# ==== RENDER TEMPLATE ====
sed \
  -e "s|{{JOB_NAME}}|$JOB_NAME|g" \
  -e "s|{{CPUS}}|$CPUS|g" \
  -e "s|{{mem}}|$mem|g" \
  -e "s|{{SCRIPT_FILE}}|$SCRIPT_FILE|g" \
  -e "s|\$SCRIPT_DIR|$SCRIPT_DIR|g" \
  -e "s|{{OUTPUT_DIR}}|$OUTPUT_DIR|g" \
  "$SCRIPT_DIR/Template.sh" > "$TEMP_JOB"

# ==== SUBMIT JOB ====
echo "Submitting job: $JOB_NAME"
sbatch "$TEMP_JOB"

echo $SCRIPT_DIR

# ==== CLEAN UP ====
rm -f "$TEMP_JOB"
 