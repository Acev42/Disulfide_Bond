#!/usr/bin/env bash
set -e

# ===================== Default Parameters =====================
INDEX=""
THRESHOLD="0.5"
INPUT="./input/"
OUTPUT="./output"
PDB=""
CHAIN=""
FILETYPE="PDB"
MODEL="./src/model/Disulfide_searching.ckpt"
SAVE="false"
IGNORE="true"

# ===================== Help =====================
show_help() {
cat <<EOF
Usage: $0 [options]

Disulfide bond prediction wrapper script.

Options:
  --index <value>        Index of the input file
  --threshold <value>    Probability threshold (default: 0.5)
  --input <path>          Input file or directory (default: ./input/)
  --output <path>         Output directory (default: ./output)
  --pdb <id>              PDB ID (optional)
  --chain <id>            Chain ID(s), comma-separated
  --filetype <PDB|CIF>     Structure file type (default: PDB)
  --model <path>           Path to pretrained model checkpoint
  --save                   Save intermediate files
  --ignore [true|false]    Ignore native disulfide bonds (default: true)
  --no-ignore              Do not ignore native disulfide bonds
  -h, --help               Show this message and exit

Examples:
  $0 --index 0 --input 1AHO.pdb
  $0 --index 0 --input 1AHO.pdb --no-ignore
EOF
}

# ===================== Parse Arguments =====================
PYTHON_ARGS=()

while [[ $# -gt 0 ]]; do
  case $1 in
    --index)      INDEX="$2";      PYTHON_ARGS+=("$1" "$2"); shift 2 ;;
    --threshold)  THRESHOLD="$2";  PYTHON_ARGS+=("$1" "$2"); shift 2 ;;
    --input)      INPUT="$2";      PYTHON_ARGS+=("$1" "$2"); shift 2 ;;
    --output)     OUTPUT="$2";     PYTHON_ARGS+=("$1" "$2"); shift 2 ;;
    --pdb)        PDB="$2";        PYTHON_ARGS+=("$1" "$2"); shift 2 ;;
    --chain)      CHAIN="$2";      PYTHON_ARGS+=("$1" "$2"); shift 2 ;;
    --filetype)
      FILETYPE="$2"
      [[ "$FILETYPE" =~ ^(PDB|CIF)$ ]] || { echo "Invalid filetype: $FILETYPE"; exit 1; }
      PYTHON_ARGS+=("$1" "$2")
      shift 2
      ;;
    --model)      MODEL="$2";      PYTHON_ARGS+=("$1" "$2"); shift 2 ;;
    --save)       SAVE="true";     PYTHON_ARGS+=("--save"); shift ;;
    --ignore)
      if [[ "$2" =~ ^(true|false|True|False)$ ]]; then
        IGNORE="$2"
        PYTHON_ARGS+=("--ignore" "$2")
        shift 2
      else
        IGNORE="true"
        PYTHON_ARGS+=("--ignore")
        shift
      fi
      ;;
    --no-ignore)  IGNORE="false";  PYTHON_ARGS+=("--ignore" "false"); shift ;;
    -h|--help)    show_help; exit 0 ;;
    *) echo "Unknown option: $1"; show_help; exit 1 ;;
  esac
done

# ===================== Check Python Script =====================
PYTHON_SCRIPT="./run.py"
[[ -f "$PYTHON_SCRIPT" ]] || { echo "Missing $PYTHON_SCRIPT"; exit 1; }

# ===================== Summary =====================
cat <<EOF
========== Running Disulfide Prediction ==========
Index     : $INDEX
Threshold : $THRESHOLD
Input     : $INPUT
Output    : $OUTPUT
PDB       : $PDB
Chain     : $CHAIN
Filetype  : $FILETYPE
Model     : $MODEL
Save      : $SAVE
Ignore    : $IGNORE
=================================================
EOF

# ===================== Execute =====================
conda run -n disulfide python "$PYTHON_SCRIPT" "${PYTHON_ARGS[@]}"
echo "Done."
