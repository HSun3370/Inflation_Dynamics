#!/bin/bash
# Regime-switching random-start search on the CBS Research Grid.
#
# Each specification's N_STARTS optimizer starts are split into blocks of BLOCK starts;
# every block is one single-core job (17 specifications x 5 blocks = 85 jobs per
# frequency at the defaults). A collector per frequency is held until that frequency's
# jobs finish, then merges the blocks and writes results/ or results_monthly/.
# Each start keeps its own seed, so the selected estimate does not depend on BLOCK.
# The collector orders regimes by variance and verifies each provisional optimum
# using additional local perturbation/refits after all blocks have been merged.
# Per-block pickles, SGE logs and generated job scripts live on /scratch.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
source "${PROJECT_ROOT}/BEGE_GARCH/cbs_grid.sh"
PY_SCRIPT="${SCRIPT_DIR}/estimate_regime_switching_models.py"

FREQS="${FREQS:-Quarterly Monthly}"
N_STARTS="${N_STARTS:-5000}"
STABILITY_REPS="${STABILITY_REPS:-20}"
BLOCK="${BLOCK:-1000}"
MODEL_IDS="${MODEL_IDS:-all}"
FORCE="${FORCE:-0}"
SUBMIT_COLLECTOR="${SUBMIT_COLLECTOR:-1}"
GRID_MEM="${GRID_MEM:-2G}"
COLLECT_MEM="${COLLECT_MEM:-4G}"
LATEST_LINK="${SCRATCH_ROOT}/RegimeSwitching/latest"
RUN_DIR="${RUN_DIR:-}"

usage() {
    cat <<USAGE
Usage:
  bash ${0} submit    Submit one job per block of starts and a held collector per frequency.
  bash ${0} status    Count finished blocks per frequency and show queued/running jobs.
  bash ${0} collect   Collect existing blocks now (interactive grid_run).

Environment overrides:
  FREQS="${FREQS}"   N_STARTS=${N_STARTS}   BLOCK=${BLOCK}   STABILITY_REPS=${STABILITY_REPS}
  MODEL_IDS=${MODEL_IDS}   # "all" or a space-separated list such as "R01 R17"
  FORCE=${FORCE}           # 1 resubmits blocks that already have a result
  SUBMIT_COLLECTOR=${SUBMIT_COLLECTOR}   GRID_MEM=${GRID_MEM}   COLLECT_MEM=${COLLECT_MEM}
  RUN_DIR=<scratch run directory>   # submit: new dated directory; status/collect: ${LATEST_LINK}
USAGE
}

results_dir() {
    case "$1" in
        Quarterly) printf '%s\n' "${SCRIPT_DIR}/results" ;;
        Monthly) printf '%s\n' "${SCRIPT_DIR}/results_monthly" ;;
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

collect_args() {
    local freq="$1" run_dir="$2"
    printf -- '--collect --raw-dir "%s" --n-starts %s --stability-reps %s --data-path "%s" --output-dir "%s"' \
        "${run_dir}/${freq}/raw" "${N_STARTS}" "${STABILITY_REPS}" "$(data_path "${freq}")" "$(results_dir "${freq}")"
}

submit_jobs() {
    if [ $((N_STARTS % BLOCK)) -ne 0 ]; then
        echo "N_STARTS=${N_STARTS} must be a multiple of BLOCK=${BLOCK}" >&2
        exit 1
    fi
    RUN_DIR="${RUN_DIR:-${SCRATCH_ROOT}/RegimeSwitching/$(date +%Y%m%d_%H%M%S)}"
    mkdir -p "${RUN_DIR}"
    ln -sfn "${RUN_DIR}" "${LATEST_LINK}"
    echo "Run directory: ${RUN_DIR} (linked from ${LATEST_LINK})"
    echo "Starts per specification: ${N_STARTS} in blocks of ${BLOCK}"

    local ids
    ids=$(model_ids)
    local freq
    for freq in ${FREQS}; do
        local raw="${RUN_DIR}/${freq}/raw" logs="${RUN_DIR}/${freq}/job-outs" scripts="${RUN_DIR}/${freq}/bash"
        mkdir -p "${raw}" "${logs}" "${scripts}"
        local job_ids=() model_id first
        for model_id in ${ids}; do
            for ((first = 1; first <= N_STARTS; first += BLOCK)); do
                local last=$((first + BLOCK - 1))
                local block_name
                block_name=$(printf '%s_%04d-%04d' "${model_id}" "${first}" "${last}")
                if [ "${FORCE}" != "1" ] && [ -f "${raw}/${block_name}.pkl" ]; then
                    continue
                fi
                local job_script="${scripts}/rs_${freq:0:1}_${block_name}.sh"
                cat > "${job_script}" <<EOF
#!/bin/bash
# Regime switching ${freq} ${model_id} starts ${first}-${last}: grid_run --grid_submit=batch --grid_ncpus=1 --grid_mem=${GRID_MEM}
set -euo pipefail

$(python_setup_block)
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

cd "${SCRIPT_DIR}"
echo "\${JOB_NAME:-} (job \${JOB_ID:-}) on \$(hostname), starts \$(date)"
start_time=\$(date +%s)

python3 -u "${PY_SCRIPT}" --model-id ${model_id} --attempts ${first} ${last} \\
    --raw-dir "${raw}" --data-path "$(data_path "${freq}")"

echo "Ends \$(date); elapsed \$(( \$(date +%s) - start_time )) s"
EOF
                local job_id
                job_id=$(grid_submit_batch "${logs}" "${job_script}" 1 "${GRID_MEM}")
                job_ids+=("${job_id}")
            done
        done
        echo "${freq}: submitted ${#job_ids[@]} block jobs (logs in ${logs})"

        if [ "${SUBMIT_COLLECTOR}" = "1" ] && [ "${#job_ids[@]}" -gt 0 ]; then
            local collector="${scripts}/rs_${freq:0:1}_collect.sh"
            cat > "${collector}" <<EOF
#!/bin/bash
# Regime switching ${freq} collector: held until the ${freq} block jobs finish.
set -euo pipefail

$(python_setup_block)
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

cd "${SCRIPT_DIR}"
echo "\${JOB_NAME:-} (job \${JOB_ID:-}) on \$(hostname), starts \$(date)"
python3 -u "${PY_SCRIPT}" $(collect_args "${freq}" "${RUN_DIR}")
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
    local n_models expected freq
    n_models=$(model_ids | wc -l)
    expected=$((n_models * N_STARTS / BLOCK))
    for freq in ${FREQS}; do
        local n
        n=$(find "${RUN_DIR}/${freq}/raw" -maxdepth 1 -name 'R*.pkl' 2>/dev/null | wc -l)
        echo "${freq}: ${n}/${expected} blocks finished"
    done
    qstat -u "$(whoami)" | awk 'NR > 2 {print $5}' | sort | uniq -c | sed 's/^/  jobs in state /'
}

collect_now() {
    RUN_DIR="${RUN_DIR:-$(readlink -f "${LATEST_LINK}")}"
    local freq
    for freq in ${FREQS}; do
        (cd "${SCRIPT_DIR}" && grid_python_now "${COLLECT_MEM}" "${PY_SCRIPT}" --collect \
            --raw-dir "${RUN_DIR}/${freq}/raw" --n-starts "${N_STARTS}" \
            --stability-reps "${STABILITY_REPS}" \
            --data-path "$(data_path "${freq}")" --output-dir "$(results_dir "${freq}")")
    done
}

case "${1:-submit}" in
    submit) submit_jobs ;;
    status) show_status ;;
    collect) collect_now ;;
    -h|--help|help) usage ;;
    *) usage >&2; exit 1 ;;
esac
