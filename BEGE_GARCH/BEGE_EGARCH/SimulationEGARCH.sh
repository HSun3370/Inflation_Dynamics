#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
source "${SCRIPT_DIR}/../cbs_grid.sh"
FREQUENCY="${FREQUENCY:-quarterly}"
RESULTS_DIR=$(bege_results_dir "${SCRIPT_DIR}" "${FREQUENCY}")
OUTPUT_DIR="${OUTPUT_DIR:-${SCRATCH_ROOT}/BEGE_GARCH/$(basename "${SCRIPT_DIR}")/output_${FREQUENCY}_$(bege_data_vintage "${PROJECT_ROOT}")}"
ACTION_NAME="BEGE_EGARCH_RandomSearch"
PY_SCRIPT="${SCRIPT_DIR}/BEGE_EGARCH1.py"
COLLECT_SCRIPT="${SCRIPT_DIR}/collect_egarch_results.py"

START_ID="${START_ID:-1}"
END_ID="${END_ID:-10}"
N_DRAWS="${N_DRAWS:-40}"
N_STARTS="${N_STARTS:-25}"
MAXITER="${MAXITER:-800}"
TOL="${TOL:-1e-8}"
INCLUDE_ARX22="${INCLUDE_ARX22:-1}"
DENSITY_HYPERU_METHOD="${DENSITY_HYPERU_METHOD:-scipy_approx}"
COMPUTE_SE="${COMPUTE_SE:-0}"
SUBMIT_COLLECTOR="${SUBMIT_COLLECTOR:-1}"

GRID_NCPUS="${GRID_NCPUS:-4}"
GRID_MEM="${GRID_MEM:-10G}"
COLLECT_MEM="${COLLECT_MEM:-4G}"

usage() {
    cat <<USAGE
Usage:
  bash ${0} submit    Submit seed jobs and a dependent collector job.
  bash ${0} collect   Merge existing raw CSV files and write markdown now (interactive grid_run).
  bash ${0} estimate  Print rough optimizer-start and wall-time estimates.

Environment overrides:
  FREQUENCY=${FREQUENCY}  # quarterly or monthly; results go to ${RESULTS_DIR}
  START_ID=${START_ID} END_ID=${END_ID}
  N_DRAWS=${N_DRAWS} N_STARTS=${N_STARTS} MAXITER=${MAXITER} TOL=${TOL}
  INCLUDE_ARX22=${INCLUDE_ARX22}  # 1 estimates all six mean processes
  DENSITY_HYPERU_METHOD=${DENSITY_HYPERU_METHOD}
  COMPUTE_SE=${COMPUTE_SE}         # 0 skips standard errors for fast search
  SUBMIT_COLLECTOR=${SUBMIT_COLLECTOR}
  OUTPUT_DIR=${OUTPUT_DIR}  # raw CSVs, SGE logs and job scripts
  GRID_NCPUS=${GRID_NCPUS} GRID_MEM=${GRID_MEM} COLLECT_MEM=${COLLECT_MEM}  # memory is the job total
  PYTHON_ENV_BIN=${PYTHON_ENV_BIN}

Default workload:
  10 seed jobs x 40 draws x 25 starts = 10,000 starts per mean process.
  With all six mean processes, this is 60,000 optimizer starts total.
  Each seed job checkpoints a CSV after every completed draw.
USAGE
}

mean_count() {
    if [ "${INCLUDE_ARX22}" = "1" ]; then
        printf '6\n'
    else
        printf '5\n'
    fi
}

estimate_runtime() {
    local jobs means starts_per_seed starts_per_mean total_starts
    jobs=$((END_ID - START_ID + 1))
    means=$(mean_count)
    starts_per_seed=$((means * N_DRAWS * N_STARTS))
    starts_per_mean=$((jobs * N_DRAWS * N_STARTS))
    total_starts=$((means * starts_per_mean))

    echo "Jobs: ${jobs}"
    echo "Mean processes per job: ${means}"
    echo "Starts per seed job: ${starts_per_seed}"
    echo "Starts per mean process: ${starts_per_mean}"
    echo "Total optimizer starts: ${total_starts}"
    echo "BEGE-EGARCH couples the two shape recursions, so it may be slower than Full BEGE."
}

elapsed_block() {
    cat <<'EOF'
end_time=$(date +%s)
elapsed=$((end_time - start_time))
printf 'Elapsed time: %d days %02d hr %02d min %02d sec\n' \
    $((elapsed / 86400)) \
    $(((elapsed % 86400) / 3600)) \
    $(((elapsed % 3600) / 60)) \
    $((elapsed % 60))
EOF
}

build_python_args() {
    local args
    args="--id PLACEHOLDER_ID --n-draws ${N_DRAWS} --n-starts ${N_STARTS} --maxiter ${MAXITER} --tol ${TOL} --density-hyperu-method ${DENSITY_HYPERU_METHOD} --output-dir '${OUTPUT_DIR}'"
    if [ "${INCLUDE_ARX22}" != "1" ]; then
        args="${args} --skip-arx22"
    fi
    if [ "${COMPUTE_SE}" = "1" ]; then
        args="${args} --compute-se"
    fi
    printf '%s\n' "${args}"
}

run_collect_now() {
    mkdir -p "${RESULTS_DIR}"
    cd "${PROJECT_ROOT}"
    START_ID="${START_ID}" END_ID="${END_ID}" BEGE_FREQUENCY="${FREQUENCY}" \
        BEGE_COLLECT_OUTPUT_DIR="${OUTPUT_DIR}" BEGE_COLLECT_RESULTS_DIR="${RESULTS_DIR}" \
        grid_python_now "${COLLECT_MEM}" "${COLLECT_SCRIPT}"
}

submit_collector() {
    local dependency="$1"
    local collector_script="${OUTPUT_DIR}/bash/${ACTION_NAME}/${ACTION_NAME}_collect.sh"

    mkdir -p "$(dirname "${collector_script}")" "${OUTPUT_DIR}/job-outs/${ACTION_NAME}/collect/"

    cat > "${collector_script}" <<EOF
#!/bin/bash
# ${ACTION_NAME} collector: grid_run --grid_submit=batch --grid_ncpus=1 --grid_mem=${COLLECT_MEM} --grid_hold=<seed jobs>

set -euo pipefail

$(python_setup_block)
export BEGE_FREQUENCY=${FREQUENCY}

cd ${PROJECT_ROOT}

echo "\${JOB_NAME:-} (job \${JOB_ID:-}) on \$(hostname)"
echo "Collector starts \$(date)"
start_time=\$(date +%s)

START_ID=${START_ID} END_ID=${END_ID} BEGE_COLLECT_OUTPUT_DIR="${OUTPUT_DIR}" BEGE_COLLECT_RESULTS_DIR="${RESULTS_DIR}" python3 -u "${COLLECT_SCRIPT}"

echo "Collector ends \$(date)"
$(elapsed_block)
EOF

    local collector_job_id
    collector_job_id=$(grid_submit_batch "${OUTPUT_DIR}/job-outs/${ACTION_NAME}/collect" "${collector_script}" 1 "${COLLECT_MEM}" "${dependency}")
    echo "Submitted collector job: ${collector_job_id}"
}

submit_jobs() {
    if [ "${START_ID}" -gt "${END_ID}" ]; then
        echo "START_ID must be <= END_ID." >&2
        exit 1
    fi

    estimate_runtime

    mkdir -p "${OUTPUT_DIR}/bash/${ACTION_NAME}" "${OUTPUT_DIR}/job-outs/${ACTION_NAME}"

    local id
    local job_ids=()
    local python_args_template
    python_args_template=$(build_python_args)

    for id in $(seq "${START_ID}" "${END_ID}"); do
        mkdir -p "${OUTPUT_DIR}/job-outs/${ACTION_NAME}/id_${id}/"
        mkdir -p "${OUTPUT_DIR}/bash/${ACTION_NAME}/id_${id}/"

        local run_script="${OUTPUT_DIR}/bash/${ACTION_NAME}/id_${id}/id_${id}.sh"
        local python_args="${python_args_template/PLACEHOLDER_ID/${id}}"

        cat > "${run_script}" <<EOF
#!/bin/bash
# ${ACTION_NAME} seed ${id}: grid_run --grid_submit=batch --grid_ncpus=${GRID_NCPUS} --grid_mem=${GRID_MEM}

set -euo pipefail

$(python_setup_block)
export BEGE_FREQUENCY=${FREQUENCY}

cd ${PROJECT_ROOT}

echo "\${JOB_NAME:-} (job \${JOB_ID:-}) on \$(hostname)"
echo "Program starts \$(date)"
start_time=\$(date +%s)

python3 -u "${PY_SCRIPT}" ${python_args}

echo "Program ends \$(date)"
$(elapsed_block)
EOF

        local job_id
        job_id=$(grid_submit_batch "${OUTPUT_DIR}/job-outs/${ACTION_NAME}/id_${id}" "${run_script}" "${GRID_NCPUS}" "${GRID_MEM}")
        job_ids+=("${job_id}")
        echo "Submitted seed id ${id}: ${job_id}"
    done
    echo "SGE logs: ${OUTPUT_DIR}/job-outs/${ACTION_NAME}/"

    if [ "${SUBMIT_COLLECTOR}" = "1" ] && [ "${#job_ids[@]}" -gt 0 ]; then
        local dependency
        dependency=$(IFS=,; echo "${job_ids[*]}")
        submit_collector "${dependency}"
    fi
}

ACTION="${1:-submit}"

case "${ACTION}" in
    submit)
        submit_jobs
        ;;
    collect)
        run_collect_now
        ;;
    estimate)
        estimate_runtime
        ;;
    -h|--help|help)
        usage
        ;;
    *)
        usage >&2
        exit 1
        ;;
esac
