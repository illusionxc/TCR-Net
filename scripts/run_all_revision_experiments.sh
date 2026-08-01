set -Eeuo pipefail


SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

JOBS="${JOBS:-2}"
QUICK="${QUICK:-0}"
FORCE="${FORCE:-0}"
INCLUDE_GENERATOR_ROBUSTNESS="${INCLUDE_GENERATOR_ROBUSTNESS:-1}"
INCLUDE_CONFIRMATORY="${INCLUDE_CONFIRMATORY:-1}"
SEEDS="${SEEDS:-42 43 44 45 46}"
TARGETED_SEEDS="${TARGETED_SEEDS:-42 43 44}"
CROSS_SEEDS="${CROSS_SEEDS:-42 43 44}"
GENERATOR_SEEDS="${GENERATOR_SEEDS:-101 202 303}"

if [[ "${QUICK}" == "1" ]]; then
  DATA_ROOT="${DATA_ROOT:-data/revision_experiments_quick}"
  OUTPUT_ROOT="${OUTPUT_ROOT:-exp_data/revision_2026_quick}"
else
  DATA_ROOT="${DATA_ROOT:-data/revision_experiments}"
  OUTPUT_ROOT="${OUTPUT_ROOT:-exp_data/revision_2026_final}"
fi
REFERENCE_ROOT="${REFERENCE_ROOT:-exp_data/revision_2026}"
FIXED_DATA_DIR="${FIXED_DATA_DIR:-${DATA_ROOT}/fixed_data}"
FIGURE_DIR="${FIGURE_DIR:-${PROJECT_ROOT}/figures_auto}"
CONFIRM_DATA_ROOT="${CONFIRM_DATA_ROOT:-data/revision_confirmatory}"
CONFIRM_OUTPUT_ROOT="${CONFIRM_OUTPUT_ROOT:-exp_data/revision_2026_confirmatory}"

find_python() {
  local candidate
  if [[ -n "${PYTHON_BIN:-}" ]]; then
    printf '%s\n' "${PYTHON_BIN}"
    return
  fi
  for candidate in \
    "$(command -v python3 2>/dev/null || true)" \
    "$(command -v python 2>/dev/null || true)"
  do
    if [[ -n "${candidate}" && -x "${candidate}" ]]; then
      if "${candidate}" -c \
        "import torch, numpy, sklearn, yaml, xgboost" >/dev/null 2>&1
      then
        printf '%s\n' "${candidate}"
        return
      fi
    fi
  done
  return 1
}

if ! PYTHON_BIN="$(find_python)"; then
  echo "[fatal] No Python executable with torch/numpy/sklearn/PyYAML/xgboost was found."
  echo "        Install requirements.txt first, or set PYTHON_BIN to a valid Python executable."
  exit 2
fi

mkdir -p "${OUTPUT_ROOT}/logs"
MASTER_LOG="${OUTPUT_ROOT}/logs/run_all_$(date '+%Y%m%d_%H%M%S').log"
exec > >(tee -a "${MASTER_LOG}") 2>&1

on_error() {
  local exit_code=$?
  echo
  echo "[failed] The experiment runner failed at line ${BASH_LINENO[0]}, exit=${exit_code}"
  echo "[failed] Main log: ${PROJECT_ROOT}/${MASTER_LOG}"
  echo "[failed] Fix the issue and rerun the same command; completed tasks will be skipped."
  exit "${exit_code}"
}
trap on_error ERR

read -r -a SEED_ARRAY <<< "${SEEDS}"
read -r -a TARGETED_SEED_ARRAY <<< "${TARGETED_SEEDS}"
read -r -a CROSS_SEED_ARRAY <<< "${CROSS_SEEDS}"
read -r -a GENERATOR_SEED_ARRAY <<< "${GENERATOR_SEEDS}"

COMMON_ARGS=(
  --config configs/tcr_net_paper.yaml
  --fixed-data-dir "${FIXED_DATA_DIR}"
  --data-root "${DATA_ROOT}"
  --output-root "${OUTPUT_ROOT}"
  --jobs "${JOBS}"
  --seeds "${SEED_ARRAY[@]}"
  --targeted-seeds "${TARGETED_SEED_ARRAY[@]}"
  --cross-seeds "${CROSS_SEED_ARRAY[@]}"
  --generator-seeds "${GENERATOR_SEED_ARRAY[@]}"
)

PREPARE_ARGS=(
  --config configs/tcr_net_paper.yaml
  --data-root "${DATA_ROOT}"
  --generator-seeds "${GENERATOR_SEED_ARRAY[@]}"
)

if [[ "${QUICK}" == "1" ]]; then
  COMMON_ARGS+=(--quick)
  PREPARE_ARGS+=(--quick)
fi
if [[ "${FORCE}" == "1" ]]; then
  COMMON_ARGS+=(--force)
  PREPARE_ARGS+=(--force)
fi

echo "============================================================"
echo "TCR-Net revised experiments"
echo "============================================================"
echo "Project root : ${PROJECT_ROOT}"
echo "Python       : ${PYTHON_BIN}"
echo "Jobs         : ${JOBS}"
echo "Quick        : ${QUICK}"
echo "Force        : ${FORCE}"
echo "Data root    : ${DATA_ROOT}"
echo "Fixed data   : ${FIXED_DATA_DIR}"
echo "Output root  : ${OUTPUT_ROOT}"
echo "Reference    : ${REFERENCE_ROOT}"
echo "Confirm data : ${CONFIRM_DATA_ROOT}"
echo "Confirm out  : ${CONFIRM_OUTPUT_ROOT}"
echo "Figure dir   : ${FIGURE_DIR}"
echo "Master log   : ${MASTER_LOG}"
echo "Seeds        : ${SEEDS}"
echo "Target seeds : ${TARGETED_SEEDS}"
echo "Cross seeds  : ${CROSS_SEEDS}"
echo

"${PYTHON_BIN}" - <<'PY'
import platform
import numpy
import sklearn
import torch
import xgboost
import yaml

print("[environment]")
print("  python  =", platform.python_version())
print("  torch   =", torch.__version__)
print("  numpy   =", numpy.__version__)
print("  sklearn =", sklearn.__version__)
print("  xgboost =", xgboost.__version__)
print("  pyyaml  =", yaml.__version__)
print("  cuda    =", torch.cuda.is_available())
PY

echo
echo "[1/12] Prepare history-window, multi-profile, and generator-seed data"
"${PYTHON_BIN}" revision_experiments.py prepare "${PREPARE_ARGS[@]}"

run_suite() {
  local suite="$1"
  echo
  echo "[suite] ${suite}"
  "${PYTHON_BIN}" revision_experiments.py run \
    --suite "${suite}" "${COMMON_ARGS[@]}"
}

echo
echo "[2/12] Main multi-seed results and strong baselines"
run_suite core

echo
echo "[3/12] Core component ablations"
run_suite ablations

echo
echo "[4/12] Bidirectional, unidirectional, and no cross-attention settings"
run_suite attention

echo
echo "[5/12] History-window K sensitivity"
run_suite history

echo
echo "[6/12] Leave-one-scenario-out cross-profile validation"
run_suite cross_scenario

if [[ "${INCLUDE_GENERATOR_ROBUSTNESS}" == "1" ]]; then
  echo
  echo "[optional] Generator-seed robustness"
  run_suite generator_robustness
fi

echo
echo "[7/12] Aggregate per-run results and mean +/- std"
"${PYTHON_BIN}" revision_experiments.py aggregate \
  --output-root "${OUTPUT_ROOT}"

echo
echo "[8/12] Generate the fair main table, paired tests, and source manifest"
"${PYTHON_BIN}" scripts/assemble_final_comparison.py \
  --original-root "${REFERENCE_ROOT}" \
  --final-root "${OUTPUT_ROOT}" \
  --fair-root "${OUTPUT_ROOT}"

echo
echo "[9/12] Generate slice metrics, confusion matrices, and error cases"
"${PYTHON_BIN}" revision_experiments.py analyze \
  --output-root "${OUTPUT_ROOT}"

echo
echo "[10/12] Generate variance, significance, and added experiment figures in the original figure layout"
"${PYTHON_BIN}" scripts/plot_revision_on_original_figures.py \
  --revision-root "${OUTPUT_ROOT}" \
  --reference-root "${REFERENCE_ROOT}" \
  --figure-dir "${FIGURE_DIR}"

if [[ "${INCLUDE_CONFIRMATORY}" == "1" ]]; then
  CONFIRM_PREP_ARGS=(
    --config configs/tcr_net_paper.yaml
    --data-root "${CONFIRM_DATA_ROOT}"
    --base-generator-seed 404
    --history-windows 16
    --scenarios transmission_inspection
    --generator-seeds 404
    --scenario-train-samples 64
    --scenario-val-samples 32
    --scenario-test-samples 64
  )
  CONFIRM_RUN_ARGS=(
    --suite core
    --config configs/tcr_net_paper.yaml
    --fixed-data-dir "${CONFIRM_DATA_ROOT}/fixed_data"
    --data-root "${CONFIRM_DATA_ROOT}"
    --output-root "${CONFIRM_OUTPUT_ROOT}"
    --seeds "${SEED_ARRAY[@]}"
    --methods TCR-Net "w/o Rule" XGBoost+Rule GRU+Rule
    --jobs "${JOBS}"
  )
  if [[ "${QUICK}" == "1" ]]; then
    CONFIRM_PREP_ARGS+=(--quick)
    CONFIRM_RUN_ARGS+=(--quick)
  fi
  if [[ "${FORCE}" == "1" ]]; then
    CONFIRM_PREP_ARGS+=(--force)
    CONFIRM_RUN_ARGS+=(--force)
  fi

  echo
  echo "[11/12] Prepare frozen confirmatory data with held-out generator seed 404"
  "${PYTHON_BIN}" revision_experiments.py prepare "${CONFIRM_PREP_ARGS[@]}"

  echo
  echo "[12/12] Run and aggregate the one-shot confirmatory evaluation"
  "${PYTHON_BIN}" revision_experiments.py run "${CONFIRM_RUN_ARGS[@]}"
  "${PYTHON_BIN}" revision_experiments.py aggregate \
    --output-root "${CONFIRM_OUTPUT_ROOT}"
fi

echo
echo "============================================================"
echo "All revised experiments finished"
echo "Per-run results : ${OUTPUT_ROOT}/aggregated/raw_results.csv"
echo "Aggregated results : ${OUTPUT_ROOT}/aggregated/aggregate_results.csv"
echo "Fair main table : ${OUTPUT_ROOT}/final_comparison/main_results.csv"
echo "Paired tests : ${OUTPUT_ROOT}/final_comparison/paired_significance.csv"
echo "Error analysis : ${OUTPUT_ROOT}/error_analysis/"
echo "Manuscript figures : ${FIGURE_DIR}/"
echo "Confirmatory results : ${CONFIRM_OUTPUT_ROOT}/aggregated/aggregate_results.csv"
echo "Main log   : ${MASTER_LOG}"
echo "============================================================"
