# Plotting for SUEP Analysis (Muon counting search)

The `runner.py` script produces `.pkl` files that contain all the histograms. These histograms are already normalized using the sum of the generator weights and the cross section of the process. The histograms are stored in a dictionary with the following structure:

```python
{
    "QCD_Pt-300To470_MuEnrichedPt5_TuneCP5_13TeV-pythia8+RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2+NANOAODSIM": {
        "muon_pt": hist.Hist(...),
        "nMuon": hist.Hist(...),
        "some_other_histogram": hist.Hist(...),
        ...
    },
    ...
}
```

All you need to load the histograms is to use pickle. A nicer wrapper function is provided in `plot_utils.py`:

```python
import plot_utils

plots = plot_utils.loader(
    tag="your_tag_here", # The tag you used in runner.py
    custom_lumi=None, # If you want to override the auto lumi calculation
    load_data=False,  # If you want to load the data histograms too
    verbosity=0, # for debugging
)
```

To facilitate producing frequently used plots, a few plotting scripts are available:

- `plot_regions.py` can be used to make plots of the `nMuon` distribution in the signal, control, and validation regions,
- `plot_systematics.py` can be used to plot systematic variations,
- `make_plots.py` will make plots that can be passed to combine.
