#!/bin/bash

#python runner.py \
#    --workflow SUEP_nbjet_comparison -o nbjet_comparison \
#    --json filelist/QCD_HT_nanoaodsim_central.json \
#    --executor dask/lpc --chunk 40000 --mild_scaleout \
#    --trigger TripleMu --era 2018 --isMC

python runner.py \
    --workflow SUEP_nbjet_comparison -o nbjet_comparison \
    --json filelist/QCD_Pt_nanoaodsim_central.json \
    --executor dask/lpc --chunk 40000 --mild_scaleout \
    --trigger TripleMu --era 2018 --isMC
