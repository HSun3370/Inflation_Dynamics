#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
OUTPUT_DIR="${SCRIPT_DIR}/output"
ACTION_NAME="BEGE_Constant_RandomSearch"
PY_SCRIPT="${SCRIPT_DIR}/BEGE_constant.py"

idarray=($(seq 1 100))

for id in "${idarray[@]}"; do
    mkdir -p "${OUTPUT_DIR}/job-outs/${ACTION_NAME}/id_${id}/"
    mkdir -p "${OUTPUT_DIR}/bash/${ACTION_NAME}/id_${id}/"

    run_script="${OUTPUT_DIR}/bash/${ACTION_NAME}/id_${id}/run.sh"

    cat > "${run_script}" <<EOF
#!/bin/bash

#SBATCH --account=pi-lhansen
#SBATCH --job-name=id_${id}
#SBATCH --output=${OUTPUT_DIR}/job-outs/${ACTION_NAME}/id_${id}/run.out
#SBATCH --error=${OUTPUT_DIR}/job-outs/${ACTION_NAME}/id_${id}/run.err
#SBATCH --time=1-11:00:00
#SBATCH --partition=caslake
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=10G

module load python/anaconda-2022.05
source ~/myenv/bin/activate

cd ${PROJECT_ROOT}

echo "\$SLURM_JOB_NAME"
echo "Program starts \$(date)"
start_time=\$(date +%s)

python3 -u "${PY_SCRIPT}" --id ${id}

echo "Program ends \$(date)"
end_time=\$(date +%s)
elapsed=\$((end_time - start_time))
eval "echo Elapsed time: \$(date -ud @\$elapsed +'\$((%s/3600/24)) days %H hr %M min %S sec')"
EOF

    sbatch "${run_script}"
done
