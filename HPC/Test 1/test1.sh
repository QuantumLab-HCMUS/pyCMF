#!/bin/bash
#SBATCH --job-name=TEST1
#SBATCH --output="/home/pnduy/pyCMF/HPC/Test 1/output.txt"
#SBATCH --error="/home/pnduy/pyCMF/HPC/Test 1/error.txt"
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=8192

# ====================================================================
# 1. KÍCH HOẠT MÔI TRƯỜNG CONDA CÁ NHÂN
# ====================================================================
# Đảm bảo lệnh conda có thể hoạt động trên compute node
source /share/installs/miniconda3/etc/profile.d/conda.sh 
conda activate py310

# Kiểm tra nhanh xem đã nhận đúng pycmf chưa (Ghi vào log output)
echo "Using python from: $(which python)"
python -c "import pycmf; print('pyCMF loaded from:', pycmf.__file__)"

# ====================================================================
# 2. CẤU HÌNH BIẾN MÔI TRƯỜNG
# ====================================================================
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export JOB_SCRATCH_PATH="/scratch/$SLURM_JOB_ID"
export TMPDIR="$JOB_SCRATCH_PATH"
export PYSCF_TMPDIR="$JOB_SCRATCH_PATH" # PySCF sẽ dùng scratch để tính tích phân cho nhanh

mkdir -p "${JOB_SCRATCH_PATH}"
RESULTS="/home/pnduy/pyCMF/HPC/Test 1/results"
mkdir -p "${RESULTS}"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S") 
PARAM_STRING="Test1"

# ====================================================================
# 3. CHẠY TÍNH TOÁN
# ====================================================================
cd "/home/pnduy/pyCMF/HPC/Test 1/" || exit 1

echo "Job started at: $(date)"

# Dùng 'python' thay cho 'python3.7' để nhận diện môi trường ảo
python calc-trans-code.py | tee "${RESULTS}/Test_output_${PARAM_STRING}_${TIMESTAMP}.txt"

echo "Job completed at: $(date)"