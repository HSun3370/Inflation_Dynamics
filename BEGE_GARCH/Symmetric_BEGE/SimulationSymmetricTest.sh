#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
source "${SCRIPT_DIR}/../cbs_grid.sh"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRATCH_ROOT}/BEGE_GARCH/$(basename "${SCRIPT_DIR}")/output}"
GRID_NCPUS="${GRID_NCPUS:-4}"
GRID_MEM="${GRID_MEM:-10G}"
SUBMIT_DIR=$(pwd)

idarray=($(seq 1 10))

hmc_python_name="SyntheticSymmetricTest.py"

action_name="Test_Symmetric_N300_Nice"

for id in "${idarray[@]}"; do

    mkdir -p "${OUTPUT_DIR}/job-outs/${action_name}/id_${id}/"
    mkdir -p "${OUTPUT_DIR}/bash/${action_name}/id_${id}/"

    run_script="${OUTPUT_DIR}/bash/${action_name}/id_${id}/id_${id}.sh"

    tee "${run_script}" <<EOF
#!/bin/bash
# ${action_name} id ${id}: grid_run --grid_submit=batch --grid_ncpus=${GRID_NCPUS} --grid_mem=${GRID_MEM}

$(python_setup_block)
export PYTHONPATH="${PROJECT_ROOT}\${PYTHONPATH:+:\${PYTHONPATH}}"
cd ${SUBMIT_DIR}

echo "\${JOB_NAME:-} (job \${JOB_ID:-}) on \$(hostname)"

echo "Program starts \$(date)"
start_time=\$(date +%s)

python3 -u ${SCRIPT_DIR}/${hmc_python_name} --id ${id}
echo "Program ends \$(date)"
end_time=\$(date +%s)
elapsed=\$((end_time - start_time))

eval "echo Elapsed time: \$(date -ud "@\$elapsed" +'\$((%s/3600/24)) days %H hr %M min %S sec')"

EOF
    job_id=$(grid_submit_batch "${OUTPUT_DIR}/job-outs/${action_name}/id_${id}" "${run_script}" "${GRID_NCPUS}" "${GRID_MEM}")
    echo "Submitted id ${id}: ${job_id}"
done
