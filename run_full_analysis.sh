#!/bin/bash -e

# Input options
# -s : signal
# -b : background

# Need pythia
source source_pythia.sh

# By default, run signal and background

all=1
signal=0
background=0
data=0
era=2018
tag=full_analysis_Jun2025
blind=1 # 0 for unblinded, 1 for blinded

while getopts 'sbde:t:' flag; do
  case "${flag}" in
    s) all=0; signal=1 ;;
    b) all=0; background=1 ;;
    d) all=0; data=1 ;;
    e) era="${OPTARG}" ;;
    t) tag="${OPTARG}" ;;
    *) echo "Unexpected option ${flag}" ;;
  esac
done

if [ $all -eq 1 ]; then
    signal=1
    background=1
    data=1
fi


# Set files according to the year
if [ "$era" = 2016 ]; then
    signal_filelist="filelists/signal/2016/GluGluToSUEP_central_UL16_May2025.json"
    background_filelist="filelists/mc_collections/2016/SUEPNano_UL16_Nov2024.json"
    data_filelist="filelists/data/2016/DoubleMuon_UL16_Nov2024.json"
elif [ "$era" = 2016APV ]; then
    signal_filelist="filelists/signal/2016APV/GluGluToSUEP_central_UL16APV_May2025.json"
    background_filelist="filelists/mc_collections/2016APV/SUEPNano_UL16APV_Nov2024.json"
    data_filelist="filelists/data/2016APV/DoubleMuon_UL16APV_Nov2024.json"
elif [ "$era" = 2017 ]; then
    signal_filelist="filelists/signal/2017/GluGluToSUEP_central_UL17_May2025.json"
    background_filelist="filelists/mc_collections/2017/SUEPNano_UL17_Nov2024.json"
    data_filelist="filelists/data/2017/DoubleMuon_UL17_Nov2024.json"
elif [ "$era" = 2018 ]; then
    signal_filelist="filelists/signal/2018/GluGluToSUEP_central_UL18_May2025.json"
    background_filelist="filelists/mc_collections/2018/SUEPNano_UL18_Nov2024.json"
    data_filelist="filelists/data/2018/DoubleMuon_UL18_Nov2024.json"
elif [ "$era" = 2022 ]; then
    signal_filelist="filelists/signal/2022/GluGluToSUEP_central_2022_May2025.json"
    background_filelist="filelists/mc_collections/2022/SUEPNano_2022_Jun2025.json"
    data_filelist="filelists/data/2022/DoubleMuon_Muon_2022_Apr2025.json"
elif [ "$era" = 2022EE ]; then
    signal_filelist="filelists/signal/2022EE/GluGluToSUEP_central_2022EE_May2025.json"
    background_filelist="filelists/mc_collections/2022EE/SUEPNano_2022EE_Jun2025.json"
    data_filelist="filelists/data/2022EE/Muon_2022EE_Apr2025.json"
elif [ "$era" = 2023 ]; then
    signal_filelist="filelists/signal/2023/GluGluToSUEP_central_2023_May2025.json"
    background_filelist="filelists/mc_collections/2023/SUEPNano_2023_Jun2025.json"
    data_filelist="filelists/data/2023/Muon0_Muon1_2023_Apr2025.json"
elif [ "$era" = 2023BPix ]; then
    signal_filelist="filelists/signal/2023BPix/GluGluToSUEP_central_2023BPix_May2025.json"
    background_filelist="filelists/mc_collections/2023BPix/SUEPNano_2023BPix_Jun2025.json"
    data_filelist="filelists/data/2023BPix/Muon0_Muon1_2023BPix_Apr2025.json"
else
    echo "Invalid era specified. Available options are:"
    echo -e "\n\t2016, 2016APV, 2017, 2018, 2022, 2022EE, 2023, or 2023BPix.\n"
    exit 1
fi


if [ $signal -eq 1 ]; then
    # echo "Processing signal SRs for ${era}..."
    # python runner.py \
    #     --workflow SUEP_coffea_SRs -o "processor_output_files/${tag}_${era}_SRs" \
    #     --json $signal_filelist --era "$era" --skimmed --isMC --do_syst \
    #     --executor futures -j "$(awk "BEGIN { print int($(nproc) * 0.15) }")" --chunk 30000
    echo "Processing signal CR for ${era}..."
    python runner.py \
        --workflow SUEP_coffea_CRs -o "processor_output_files/${tag}_${era}_CR" \
        --json $signal_filelist --era "$era" --skimmed --isMC --do_syst \
        --executor futures -j "$(awk "BEGIN { print int($(nproc) * 0.6) }")" --chunk 40000
    # echo "Processing signal VR for ${era}..."
    # python runner.py \
    #     --workflow SUEP_coffea_VR -o "processor_output_files/${tag}_${era}_VR" \
    #     --json $signal_filelist --era "$era" --skimmed --isMC --do_syst \
    #     --executor futures -j "$(awk "BEGIN { print int($(nproc) * 0.7) }")" --chunk 70000
fi

if [ $background -eq 1 ]; then
    # echo "Processing BKG SRs for ${era}..."
    # python runner.py \
    #     --workflow SUEP_coffea_SRs -o "processor_output_files/${tag}_${era}_SRs" \
    #     --json $background_filelist --era "$era" --skimmed --isMC --do_syst \
    #     --executor futures -j "$(awk "BEGIN { print int($(nproc) * 0.17) }")" --chunk 35000
    echo "Processing BKG CR for ${era}..."
    python runner.py \
        --workflow SUEP_coffea_CRs -o "processor_output_files/${tag}_${era}_CR" \
        --json $background_filelist --era "$era" --skimmed --isMC --do_syst \
        --executor futures -j "$(awk "BEGIN { print int($(nproc) * 0.55) }")" --chunk 70000
    # echo "Processing BKG VR for ${era}..."
    # python runner.py \
    #     --workflow SUEP_coffea_VR -o "processor_output_files/${tag}_${era}_VR" \
    #     --json $background_filelist --era "$era" --skimmed --isMC --do_syst \
    #     --executor futures -j "$(awk "BEGIN { print int($(nproc) * 0.7) }")" --chunk 70000
fi

if [ $data -eq 1 ]; then
    if [ $blind -eq 0 ]; then
        echo "Processing data SR for ${era}..."
        python runner.py \
            --workflow SUEP_coffea_SRs -o "processor_output_files/${tag}_${era}_SRs" --json $data_filelist \
            --era "$era" --executor futures -j "$(awk "BEGIN { print int($(nproc) * 0.84) }")" --chunk 4000
    fi
    echo "Processing data CR for ${era}..."
    python runner.py \
        --workflow SUEP_coffea_CRs -o "processor_output_files/${tag}_${era}_CR" --json $data_filelist \
        --era "$era" --executor futures -j "$(awk "BEGIN { print int($(nproc) * 0.8) }")" --chunk 70000
    # echo "Processing data VR for ${era}..."
    # python runner.py \
    #     --workflow SUEP_coffea_VR -o "processor_output_files/${tag}_${era}_VR" \
    #     --json $data_filelist --era "$era" --executor futures \
    #     -j "$(awk "BEGIN { print int($(nproc) * 0.8) }")" --chunk 70000
fi
