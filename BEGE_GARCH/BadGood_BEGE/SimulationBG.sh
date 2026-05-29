#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
OUTPUT_DIR="${SCRIPT_DIR}/output"
ACTION_NAME="BadGood_BEGE_RandomSearch"
PY_SCRIPT="${SCRIPT_DIR}/BG_GJR1.py"
COLLECT_SCRIPT="${SCRIPT_DIR}/collect_bg_results.py"

START_ID="${START_ID:-1}"
END_ID="${END_ID:-400}"
N_DRAWS="${N_DRAWS:-1}"
N_STARTS="${N_STARTS:-25}"
MAXITER="${MAXITER:-800}"
TOL="${TOL:-1e-8}"
INCLUDE_ARX22="${INCLUDE_ARX22:-1}"
SUBMIT_COLLECTOR="${SUBMIT_COLLECTOR:-1}"
COLLECT_DEPENDENCY_TYPE="${COLLECT_DEPENDENCY_TYPE:-afterany}"

SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-pi-lhansen}"
SBATCH_PARTITION="${SBATCH_PARTITION:-caslake}"
SBATCH_TIME="${SBATCH_TIME:-1-11:00:00}"
SBATCH_CPUS="${SBATCH_CPUS:-4}"
SBATCH_MEM="${SBATCH_MEM:-10G}"
COLLECT_TIME="${COLLECT_TIME:-02:00:00}"
COLLECT_MEM="${COLLECT_MEM:-4G}"

usage() {
    cat <<USAGE
Usage:
  bash ${0} submit    Submit seed jobs and a dependent collector job.
  bash ${0} collect   Merge existing raw CSV files and write markdown now.
  bash ${0} estimate  Print rough optimizer-start and wall-time estimates.

Environment overrides:
  START_ID=${START_ID} END_ID=${END_ID}
  N_DRAWS=${N_DRAWS} N_STARTS=${N_STARTS} MAXITER=${MAXITER} TOL=${TOL}
  INCLUDE_ARX22=${INCLUDE_ARX22}  # 1 estimates all four mean processes
  SUBMIT_COLLECTOR=${SUBMIT_COLLECTOR}

Default workload:
  400 seed jobs x 1 draw x 25 starts = 10,000 starts per mean process.
  With all four mean processes, this is 40,000 optimizer starts total.
  Each seed job writes one CSV only after it finishes.
USAGE
}

mean_count() {
    if [ "${INCLUDE_ARX22}" = "1" ]; then
        printf '4\n'
    else
        printf '3\n'
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
    awk -v starts="${starts_per_seed}" 'BEGIN {
        seconds_per_start = 21287 / 8000;
        seconds = starts * seconds_per_start;
        printf "Rough per-seed lower-bound from Constant_BEGE timing: %.2f hours\n", seconds / 3600;
        printf "BadGood uses a larger SLSQP problem, so actual time may be materially longer.\n";
    }'
}

python_setup_block() {
    cat <<'EOF'
if command -v module >/dev/null 2>&1; then
    module load python/anaconda-2022.05
fi
if [ -f "${HOME}/myenv/bin/activate" ]; then
    source "${HOME}/myenv/bin/activate"
fi
EOF
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
    args="--id PLACEHOLDER_ID --n-draws ${N_DRAWS} --n-starts ${N_STARTS} --maxiter ${MAXITER} --tol ${TOL}"
    if [ "${INCLUDE_ARX22}" != "1" ]; then
        args="${args} --skip-arx22"
    fi
    printf '%s\n' "${args}"
}

run_collect_now() {
    mkdir -p "${SCRIPT_DIR}/results"
    cd "${PROJECT_ROOT}"
    python3 -u "${COLLECT_SCRIPT}"
}

submit_collector() {
    local dependency="$1"
    local collector_script="${OUTPUT_DIR}/bash/${ACTION_NAME}/collect_results.sh"

    mkdir -p "$(dirname "${collector_script}")" "${OUTPUT_DIR}/job-outs/${ACTION_NAME}/collect/"

    cat > "${collector_script}" <<EOF
#!/bin/bash
#SBATCH --account=${SBATCH_ACCOUNT}
#SBATCH --job-name=${ACTION_NAME}_collect
#SBATCH --output=${OUTPUT_DIR}/job-outs/${ACTION_NAME}/collect/run.out
#SBATCH --error=${OUTPUT_DIR}/job-outs/${ACTION_NAME}/collect/run.err
#SBATCH --time=${COLLECT_TIME}
#SBATCH --partition=${SBATCH_PARTITION}
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=${COLLECT_MEM}
#SBATCH --dependency=${COLLECT_DEPENDENCY_TYPE}:${dependency}

set -euo pipefail

$(python_setup_block)

cd ${PROJECT_ROOT}

echo "\$SLURM_JOB_NAME"
echo "Collector starts \$(date)"
start_time=\$(date +%s)

python3 -u "${COLLECT_SCRIPT}"

echo "Collector ends \$(date)"
$(elapsed_block)
EOF

    local collector_job_id
    collector_job_id=$(sbatch --parsable "${collector_script}")
    echo "Submitted collector job: ${collector_job_id%%;*}"
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

        local run_script="${OUTPUT_DIR}/bash/${ACTION_NAME}/id_${id}/run.sh"
        local python_args="${python_args_template/PLACEHOLDER_ID/${id}}"

        cat > "${run_script}" <<EOF
#!/bin/bash
#SBATCH --account=${SBATCH_ACCOUNT}
#SBATCH --job-name=id_${id}
#SBATCH --output=${OUTPUT_DIR}/job-outs/${ACTION_NAME}/id_${id}/run.out
#SBATCH --error=${OUTPUT_DIR}/job-outs/${ACTION_NAME}/id_${id}/run.err
#SBATCH --time=${SBATCH_TIME}
#SBATCH --partition=${SBATCH_PARTITION}
#SBATCH --nodes=1
#SBATCH --cpus-per-task=${SBATCH_CPUS}
#SBATCH --mem=${SBATCH_MEM}

set -euo pipefail

$(python_setup_block)

cd ${PROJECT_ROOT}

echo "\$SLURM_JOB_NAME"
echo "Program starts \$(date)"
start_time=\$(date +%s)

python3 -u "${PY_SCRIPT}" ${python_args}

echo "Program ends \$(date)"
$(elapsed_block)
EOF

        local job_id_raw
        local job_id
        job_id_raw=$(sbatch --parsable "${run_script}")
        job_id="${job_id_raw%%;*}"
        job_ids+=("${job_id}")
        echo "Submitted seed id ${id}: ${job_id}"
    done

    if [ "${SUBMIT_COLLECTOR}" = "1" ] && [ "${#job_ids[@]}" -gt 0 ]; then
        local dependency
        dependency=$(IFS=:; echo "${job_ids[*]}")
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
