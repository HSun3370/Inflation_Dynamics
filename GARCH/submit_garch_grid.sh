#!/bin/bash
# GARCH-family random-start search on the CBS Research Grid.
#
# One single-core job per model combination (3 distributions x 6 mean processes x
# 12 volatility specifications = 216 jobs per frequency), each running N_STARTS
# optimizer starts, plus one collector per frequency that is held until that
# frequency's jobs finish and then writes results_garch_distributions*/.
# Per-combination pickles, SGE logs and generated job scripts live on /scratch.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
source "${PROJECT_ROOT}/BEGE_GARCH/cbs_grid.sh"
PY_SCRIPT="${SCRIPT_DIR}/estimate_garch_results.py"

FREQS="${FREQS:-Quarterly Monthly}"
N_STARTS="${N_STARTS:-5000}"
MAXITER="${MAXITER:-8000}"
SEED="${SEED:-20260622}"
MODEL_IDS="${MODEL_IDS:-all}"
FORCE="${FORCE:-0}"
SUBMIT_COLLECTOR="${SUBMIT_COLLECTOR:-1}"
GRID_MEM="${GRID_MEM:-1G}"
COLLECT_MEM="${COLLECT_MEM:-4G}"
LATEST_LINK="${SCRATCH_ROOT}/GARCH/latest"
RUN_DIR="${RUN_DIR:-}"

usage() {
    cat <<USAGE
Usage:
  bash ${0} submit    Submit one job per model combination and a held collector per frequency.
  bash ${0} status    Count finished combinations per frequency and show queued/running jobs.
  bash ${0} collect   Collect existing per-combination results now (interactive grid_run).

Environment overrides:
  FREQS="${FREQS}"   N_STARTS=${N_STARTS}   MAXITER=${MAXITER}   SEED=${SEED}
  MODEL_IDS=${MODEL_IDS}   # "all" or a space-separated list such as "M001 M050"
  FORCE=${FORCE}           # 1 resubmits combinations that already have a result
  SUBMIT_COLLECTOR=${SUBMIT_COLLECTOR}   GRID_MEM=${GRID_MEM}   COLLECT_MEM=${COLLECT_MEM}
  RUN_DIR=<scratch run directory>   # submit: new dated directory; status/collect: ${LATEST_LINK}
USAGE
}

results_dir() {
    case "$1" in
        Quarterly) printf '%s\n' "${SCRIPT_DIR}/results_garch_distributions" ;;
        Monthly) printf '%s\n' "${SCRIPT_DIR}/results_garch_distributions_monthly" ;;
        *) echo "Unknown frequency: $1" >&2; return 1 ;;
    esac
}

data_path() {
    printf '%s\n' "${PROJECT_ROOT}/DataSummary/Aggregate_CPI_inflation_$1.pkl"
}

model_ids() {
    if [ "${MODEL_IDS}" = "all" ]; then
        "${PYTHON_ENV_BIN}/python3" "${PY_SCRIPT}" --list-models | cut -f1
    else
        printf '%s\n' ${MODEL_IDS}
    fi
}

collector_command() {
    local freq="$1" run_dir="$2"
    printf 'python3 -u "%s" --collect --raw-dir "%s" --data-path "%s" --output-dir "%s"' \
        "${PY_SCRIPT}" "${run_dir}/${freq}/raw" "$(data_path "${freq}")" "$(results_dir "${freq}")"
}

submit_jobs() {
    RUN_DIR="${RUN_DIR:-${SCRATCH_ROOT}/GARCH/$(date +%Y%m%d_%H%M%S)}"
    mkdir -p "${RUN_DIR}"
    ln -sfn "${RUN_DIR}" "${LATEST_LINK}"
    echo "Run directory: ${RUN_DIR} (linked from ${LATEST_LINK})"
    echo "Starts per combination: ${N_STARTS}"

    local ids
    ids=$(model_ids)
    local freq
    for freq in ${FREQS}; do
        local raw="${RUN_DIR}/${freq}/raw" logs="${RUN_DIR}/${freq}/job-outs" scripts="${RUN_DIR}/${freq}/bash"
        mkdir -p "${raw}" "${logs}" "${scripts}"
        local job_ids=() model_id
        for model_id in ${ids}; do
            if [ "${FORCE}" != "1" ] && [ -f "${raw}/${model_id}.pkl" ]; then
                continue
            fi
            local job_script="${scripts}/garch_${freq:0:1}_${model_id}.sh"
            cat > "${job_script}" <<EOF
#!/bin/bash
# GARCH ${freq} ${model_id}: grid_run --grid_submit=batch --grid_ncpus=1 --grid_mem=${GRID_MEM}
set -euo pipefail

$(python_setup_block)
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

cd "${SCRIPT_DIR}"
echo "\${JOB_NAME:-} (job \${JOB_ID:-}) on \$(hostname), starts \$(date)"
start_time=\$(date +%s)

python3 -u "${PY_SCRIPT}" --model-id ${model_id} --raw-dir "${raw}" \\
    --data-path "$(data_path "${freq}")" --n-starts ${N_STARTS} --maxiter ${MAXITER} --seed ${SEED}

echo "Ends \$(date); elapsed \$(( \$(date +%s) - start_time )) s"
EOF
            local job_id
            job_id=$(grid_submit_batch "${logs}" "${job_script}" 1 "${GRID_MEM}")
            job_ids+=("${job_id}")
        done
        echo "${freq}: submitted ${#job_ids[@]} combination jobs (logs in ${logs})"

        if [ "${SUBMIT_COLLECTOR}" = "1" ] && [ "${#job_ids[@]}" -gt 0 ]; then
            local collector="${scripts}/garch_${freq:0:1}_collect.sh"
            cat > "${collector}" <<EOF
#!/bin/bash
# GARCH ${freq} collector: held until the ${freq} combination jobs finish.
set -euo pipefail

$(python_setup_block)

cd "${SCRIPT_DIR}"
echo "\${JOB_NAME:-} (job \${JOB_ID:-}) on \$(hostname), starts \$(date)"
$(collector_command "${freq}" "${RUN_DIR}")
EOF
            local hold collector_id
            hold=$(IFS=,; echo "${job_ids[*]}")
            collector_id=$(grid_submit_batch "${logs}" "${collector}" 1 "${COLLECT_MEM}" "${hold}")
            echo "${freq}: collector job ${collector_id} writes $(results_dir "${freq}")"
        fi
    done
}

show_status() {
    RUN_DIR="${RUN_DIR:-$(readlink -f "${LATEST_LINK}")}"
    echo "Run directory: ${RUN_DIR}"
    local freq total
    total=$("${PYTHON_ENV_BIN}/python3" "${PY_SCRIPT}" --list-models | wc -l)
    for freq in ${FREQS}; do
        local n
        n=$(find "${RUN_DIR}/${freq}/raw" -maxdepth 1 -name 'M*.pkl' 2>/dev/null | wc -l)
        echo "${freq}: ${n}/${total} combinations finished"
    done
    qstat -u "$(whoami)" | awk 'NR > 2 {print $5}' | sort | uniq -c | sed 's/^/  jobs in state /'
}

collect_now() {
    RUN_DIR="${RUN_DIR:-$(readlink -f "${LATEST_LINK}")}"
    local freq
    for freq in ${FREQS}; do
        (cd "${SCRIPT_DIR}" && grid_python_now "${COLLECT_MEM}" "${PY_SCRIPT}" --collect \
            --raw-dir "${RUN_DIR}/${freq}/raw" --data-path "$(data_path "${freq}")" \
            --output-dir "$(results_dir "${freq}")")
    done
}

case "${1:-submit}" in
    submit) submit_jobs ;;
    status) show_status ;;
    collect) collect_now ;;
    -h|--help|help) usage ;;
    *) usage >&2; exit 1 ;;
esac
