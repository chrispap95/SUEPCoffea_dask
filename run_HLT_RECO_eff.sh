#!/bin/bash -e

source source_pythia.sh

era=2018
tag=HLT_RECO_eff_Apr2026
samplejson=""
is_mc=1

while getopts 'de:t:f:c:' flag; do
  case "${flag}" in
    d) is_mc=0 ;;
    e) era="${OPTARG}" ;;
    t) tag="${OPTARG}" ;;
    f) samplejson="${OPTARG}" ;;
    *) echo "Unexpected option ${flag}" ;;
  esac
done

if [ -z "${samplejson}" ]; then
    if [ "${is_mc}" -eq 1 ]; then
        case "${era}" in
            2018) samplejson="filelists/unskimmed_samples/qcd_2018_local_nano.json" ;;
            *)
                echo "No default MC sample JSON is configured for era ${era}."
                echo "Pass one explicitly with -f."
                exit 1
                ;;
        esac
    else
        case "${era}" in
            2018) samplejson="filelists/unskimmed_samples/JetHT_2018_local_nano.json" ;;
            *)
                echo "No default JetHT data JSON is configured for era ${era}."
                echo "Pass one explicitly with -f."
                exit 1
                ;;
        esac
    fi
fi

cmd=(
    python runner.py
    --workflow SUEP_coffea_HLT_RECO_eff
    -o "processor_output_files/${tag}_${era}"
    --json "${samplejson}"
    --era "${era}"
    --executor dask/lpc --mild-scaleout --max-scaleout 150 --memory 2GB --chunk 100000
)

if [ "${is_mc}" -eq 1 ]; then
    cmd+=(--isMC)
fi

"${cmd[@]}"
