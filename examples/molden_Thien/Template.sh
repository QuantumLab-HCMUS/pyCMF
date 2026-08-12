#!/bin/bash
#SBATCH --job-name={{JOB_NAME}}
#SBATCH --output={{OUTPUT_DIR}}/output_%j_%x.txt
#SBATCH --error={{OUTPUT_DIR}}/error_%j_%x.txt
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={{CPUS}}
#SBATCH --mem-per-cpu={{mem}}



# Load môi trường
source ~/venvs/bin/activate

# (Không cần export nếu không dùng)
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
export PYTHONPATH="$ROOT_DIR:$PYTHONPATH"

echo PYTHONPATH: $PYTHONPATH
echo File: {{SCRIPT_FILE}}
echo Local: $SCRIPT_DIR

mole=$MOLE
start=$START
end=$END
dR=$DR

# Băt đâu đêm giờ
start=$(date +%s)

# Chạy job
cd $SCRIPT_DIR
python {{SCRIPT_FILE}}

# Xóa fie tạm
rm -rf tmp/* *.tmp

# Kêt thúc đêm giờ
end=$(date +%s)

runtime=$((end - start))

hours=$((runtime / 3600))
minutes=$(((runtime % 3600) / 60))
seconds=$((runtime % 60))

printf "Thời gian: %02d:%02d:%02d (giờ:phút:giây)\n" $hours $minutes $seconds
