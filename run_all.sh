#!/usr/bin/env bash
# Runs the full pipeline in order. Set UKB_DATA_DIR, UKB_PRS_FILE and PPDM_OUTPUT_DIR first (see README.md).
set -euo pipefail
cd "$(dirname "$0")/scripts"
for s in 01_load.py 01b_extra_codes.py 01c_opcs.py 02_main.py 03_sensitivity.py; do echo "== $s"; python3 "$s"; done
echo "== 04_finegray.R"; Rscript 04_finegray.R
for s in 05_conventional_matching.py 06_additional.py 06b_continuous.py 07_severity.py 08_extensions.py 09_early_window.py \
         10_gp_linked.py 11_prs.py 12_table1.py 13_figures.py 14_graphical_abstract.py 15_supplement.py; do echo "== $s"; python3 "$s"; done
