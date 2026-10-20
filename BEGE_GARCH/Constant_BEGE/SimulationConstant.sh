#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
source "${SCRIPT_DIR}/../cbs_grid.sh"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRATCH_ROOT}/BEGE_GARCH/$(basename "${SCRIPT_DIR}")/output}"
ACTION_NAME="BEGE_Constant_RandomSearch"
PY_SCRIPT="${SCRIPT_DIR}/BEGE_constant.py"
COLLECT_SCRIPT="${SCRIPT_DIR}/collect_constant_results.py"

START_ID="${START_ID:-1}"
END_ID="${END_ID:-50}"
N_DRAWS="${N_DRAWS:-40}"
N_STARTS="${N_STARTS:-25}"
MAXITER="${MAXITER:-1500}"
TOL="${TOL:-1e-8}"
INCLUDE_ARX22="${INCLUDE_ARX22:-1}"
SUBMIT_COLLECTOR="${SUBMIT_COLLECTOR:-1}"

GRID_NCPUS="${GRID_NCPUS:-2}"
GRID_MEM="${GRID_MEM:-10G}"
COLLECT_MEM="${COLLECT_MEM:-4G}"

usage() {
    cat <<USAGE
Usage:
  bash ${0} submit    Submit seed jobs and a dependent collector job.
  bash ${0} collect   Merge existing raw CSV files and write markdown now (interactive grid_run).

Environment overrides:
  START_ID=${START_ID} END_ID=${END_ID}
  N_DRAWS=${N_DRAWS} N_STARTS=${N_STARTS} MAXITER=${MAXITER} TOL=${TOL}
  INCLUDE_ARX22=${INCLUDE_ARX22}  # 1 reports all four mean processes
  SUBMIT_COLLECTOR=${SUBMIT_COLLECTOR}
  OUTPUT_DIR=${OUTPUT_DIR}  # raw CSVs, SGE logs and job scripts
  GRID_NCPUS=${GRID_NCPUS} GRID_MEM=${GRID_MEM} COLLECT_MEM=${COLLECT_MEM}  # memory is the job total
  PYTHON_ENV_BIN=${PYTHON_ENV_BIN}
USAGE
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
    args="--id PLACEHOLDER_ID --n-draws ${N_DRAWS} --n-starts ${N_STARTS} --maxiter ${MAXITER} --tol ${TOL} --output-dir '${OUTPUT_DIR}'"
    if [ "${INCLUDE_ARX22}" != "1" ]; then
        args="${args} --skip-arx22"
    fi
    printf '%s\n' "${args}"
}

run_collect_now() {
    mkdir -p "${SCRIPT_DIR}/results"
    cd "${PROJECT_ROOT}"
    START_ID="${START_ID}" END_ID="${END_ID}" BEGE_COLLECT_OUTPUT_DIR="${OUTPUT_DIR}" \
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

cd ${PROJECT_ROOT}

echo "\${JOB_NAME:-} (job \${JOB_ID:-}) on \$(hostname)"
echo "Collector starts \$(date)"
start_time=\$(date +%s)

START_ID=${START_ID} END_ID=${END_ID} BEGE_COLLECT_OUTPUT_DIR="${OUTPUT_DIR}" python3 -u "${COLLECT_SCRIPT}"

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
    -h|--help|help)
        usage
        ;;
    *)
        usage >&2
        exit 1
        ;;
esac
