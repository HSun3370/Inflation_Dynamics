#!/bin/bash
# Scheduler helpers for the Columbia CBS Research Grid, sourced by the BEGE submit scripts.
#
# The Grid runs Sun Grid Engine behind the grid_run wrapper (CBS_ResearchComputing_2023.md):
#   grid_run --grid_submit=batch --grid_ncpus=N --grid_mem=M [--grid_hold=IDS] /abs/path/job.sh
# becomes "qsub -cwd -V -b y -pe threaded N -l h_vmem=M/N ...", which means:
#   * #$ directives inside job scripts are ignored, so every resource request goes on the command line;
#   * --grid_mem is the TOTAL memory of the job, split evenly across its cores;
#   * the job name is the basename of the job script;
#   * SGE writes <job name>.o<job id> / .e<job id> into the directory grid_run is called from;
#   * --grid_hold takes comma-separated job ids and releases once they finish, whatever their exit status;
#   * regular jobs may run up to 480 hours, so there is no wall-time request.

# Directory whose python3 runs the jobs: the private env built from requirements.txt with
#   conda create -p ~/.conda/envs/inflation_dynamics -c conda-forge --override-channels python=3.12 pip
#   ~/.conda/envs/inflation_dynamics/bin/python -m pip install -r requirements.txt
PYTHON_ENV_BIN="${PYTHON_ENV_BIN:-${HOME}/.conda/envs/inflation_dynamics/bin}"

# Intermediate output (raw per-seed CSVs, job logs, generated job scripts) lives on /scratch,
# which has no backup and purges old files; collected results stay under each model's results/.
SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/$(whoami)/Inflation_Dynamics}"

python_setup_block() {
    cat <<EOF
export PATH="${PYTHON_ENV_BIN}:\${PATH}"
echo "Python: \$(command -v python3) (\$(python3 --version 2>&1)), NSLOTS=\${NSLOTS:-unset}"
EOF
}

# grid_submit_batch LOG_DIR JOB_SCRIPT NCPUS MEM [HOLD_JOB_IDS]
# Submits JOB_SCRIPT as a batch job from LOG_DIR and prints the SGE job id.
grid_submit_batch() {
    local log_dir="$1"
    local job_script="$2"
    local ncpus="$3"
    local mem="$4"
    local hold="${5:-}"
    local grid_args=(--grid_submit=batch "--grid_ncpus=${ncpus}" "--grid_mem=${mem}")
    if [ -n "${hold}" ]; then
        grid_args+=("--grid_hold=${hold}")
    fi

    if [ ! -x "${PYTHON_ENV_BIN}/python3" ]; then
        printf 'No python3 in PYTHON_ENV_BIN=%s; build the env from requirements.txt first.\n' "${PYTHON_ENV_BIN}" >&2
        return 1
    fi

    chmod +x "${job_script}"
    local reply
    if ! reply=$(cd "${log_dir}" && grid_run "${grid_args[@]}" "${job_script}" 2>&1); then
        printf 'grid_run failed for %s:\n%s\n' "${job_script}" "${reply}" >&2
        return 1
    fi

    local job_id
    job_id=$(printf '%s\n' "${reply}" | sed -n 's/^Your job \([0-9][0-9]*\) .*has been submitted.*/\1/p')
    if [ -z "${job_id}" ]; then
        printf 'Could not read a job id from the grid_run reply for %s:\n%s\n' "${job_script}" "${reply}" >&2
        return 1
    fi
    printf '%s\n' "${job_id}"
}

# grid_python_now MEM SCRIPT [ARGS...]
# Runs a Python script now on a worker node (interactive grid_run) instead of the login node.
grid_python_now() {
    local mem="$1"
    shift
    grid_run "--grid_mem=${mem}" "${PYTHON_ENV_BIN}/python3" -u "$@"
}
