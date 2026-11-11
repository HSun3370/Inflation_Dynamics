#!/bin/bash
# Submit the BEGE random search for several specifications and frequencies on the CBS Grid.
#
# Calls each specification's Simulation*.sh with FREQUENCY set, so every specification gets its
# seed jobs (START_ID..END_ID, 10 by default: 10 x 40 draws x 25 starts = 10,000 starts per
# mean process) and a held collector that writes results/ (quarterly) or results_monthly/
# (monthly). Other environment overrides (START_ID, END_ID, N_DRAWS, ...) pass through.
#
#   bash submit_bege_grid.sh                      # all specifications, quarterly and monthly
#   SPECS="Full_BEGE" FREQUENCIES=monthly bash submit_bege_grid.sh
set -euo pipefail

BEGE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Specification folder -> its submit script. The canonical five (AGENTS.md) plus the
# Constant-p and Constant-n restrictions reported alongside them; EGARCH BEGE is an
# extension run only on request.
declare -A SUBMIT_SCRIPT=(
    [Constant_BEGE]=SimulationConstant.sh
    [Symmetric_BEGE]=SimulationSymmetric.sh
    [BadGood_BEGE]=SimulationBG.sh
    [InflationDeflation_BEGE]=SimulationID.sh
    [Full_BEGE]=SimulationGJR.sh
    [ConstantP_BEGE]=SimulationConstantP.sh
    [ConstantN_BEGE]=SimulationConstantN.sh
    [EGARCH_BEGE]=SimulationEGARCH.sh
)
SPECS="${SPECS:-Constant_BEGE Symmetric_BEGE BadGood_BEGE InflationDeflation_BEGE Full_BEGE ConstantP_BEGE ConstantN_BEGE}"
FREQUENCIES="${FREQUENCIES:-quarterly monthly}"

for frequency in ${FREQUENCIES}; do
    for spec in ${SPECS}; do
        script="${SUBMIT_SCRIPT[${spec}]:-}"
        if [ -z "${script}" ]; then
            echo "Unknown specification ${spec}; choose from: ${!SUBMIT_SCRIPT[*]}" >&2
            exit 1
        fi
        echo "=== ${spec} (${frequency})"
        FREQUENCY="${frequency}" bash "${BEGE_DIR}/${spec}/${script}" submit
    done
done
