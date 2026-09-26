# SUEP Coffea - Muon Counting Search

This repository contains the code for the Muon Counting Search for Soft Unclustered Energy Patterns (SUEP) using coffea and dask.

## Setup

The code was devolopped for use at the LPC cluster. In principle, you should be able to run it on lxplus or any other machine, but you will need to configure the dask setup accordingly (if you want to use dask).
For use at the LPC cluster, the first time do the following:

```bash
ssh <username>@cmslpc-el9.fnal.gov
cd directory/where/you/want/to/work
git clone git@github.com:chrispap95/SUEPCoffea_dask.git
cd SUEPCoffea_dask
source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh
python -m venv myenv
pip install -r requirements.txt
```

Then, for every new session, do:

```bash
ssh <username>@cmslpc-el9.fnal.gov
cd directory/where/you/want/to/work/SUEPCoffea_dask
source setup.sh
```

It's always handy to have a valid certificate proxy:

```bash
voms-proxy-init -voms cms -rfc -valid 192:00
```

If you want to open the jupyter notebooks, you can do the following instead:

```bash
ssh -L localhost:8989:localhost:8989 <username>@cmslpc-el9.fnal.gov
cd directory/where/you/want/to/work/SUEPCoffea_dask
source jupy.sh
```

Then, just open the link that appears in the terminal in your browser.

## Running the code

It is suggested to run the code using [lpcjobqueue](https://github.com/CoffeaTeam/lpcjobqueue/) with one of the pre-packaged singularity images. This will allow you to run the code on the LPC cluster using dask. To do this, you need to have a valid grid certificate and to have the lpcjobqueue installed. Starting with a fresh environment (no `source setup.sh` or `source jupy.sh`), you can install it by running:

```bash
curl -OL https://raw.githubusercontent.com/CoffeaTeam/lpcjobqueue/main/bootstrap.sh
bash bootstrap.sh
```

Then, enter a singularity image using:

```bash
./shell coffeateam/coffea-base:0.7.22-py3.10
```

or just do `./shell` to see info about the available images.

Then, you can run the code using:

```bash
python runner.py --workflow SUEP_coffea_SR_high_temp \
    -o processor_output_files/<some_identifying_tag>_SR_high_temp \
    --json filelists/signal/GluGluToSUEP_central_UL18_Nov2024.json --era 2018 \
    --executor futures -j 8 --chunk 50000 --skimmed --trigger TripleMu --isMC
```

This will run the code for the the high temperature SR for the centrally produced signal files for 2018. The output will be saved in the `processor_output_files` directory with the tag you specified. The `--skimmed` flag is used to indicate that the input files are already skimmed. The `--trigger` flag is used to specify the trigger to be used. The `--isMC` flag is used to indicate that the input files are MC files. The code will be executed locally using 8 cores and will process the events in chunks of 50000.

To run the entire analysis, you can use the following script:

```bash
./run_full_analysis.sh -s -b
```

When using the `--executor dask/lpc` option with the `runner.py` script, dask will be used to submit jobs to condor. In the `run_full_analysis.sh` script, dask is used for the processing of the low temperature SR. Dask offers a nice dashboard to monitor the execution progress. To access it, you need forward port 44890 from the same machine:

```bash
ssh -L localhost:44890:localhost:44890 <username>@cmslpc-el9.fnal.gov
cd directory/where/you/want/to/work/SUEPCoffea_dask
./shell coffeateam/coffea-base:0.7.22-py3.10
./run_full_analysis.sh -s -b
```

Then, open a browser and go to `localhost:44890` (works only when processing `SR_low_temp`). If you forgot to forward the port, you can also forward it after the job has started but you need to ssh & forward to the same LPC machine.

## Plotting

To find out more about plotting and exporting the plots, see the `plotting` directory.

### NanoAOD event displays

`plotting.event_display` makes a compact eta-phi display of the jets, muons,
electrons, and photons in one NanoAOD event. In a notebook:

```python
from plotting.event_display import display_event

source = "root://cmseos.fnal.gov//store/path/to/file.root"
fig, ax, event = display_event(source, entry=42)
```

The returned `event` keeps the selected arrays for further inspection, for
example `event["Muon"].pt`. Muons always satisfy the baseline cleaning:
`mediumId`, $p_T>5$ GeV, $|\eta|<2.4$, and $|d_z|<0.2$ cm. At least three
cleaned muons are required in every displayed event. Object choices and
additional transverse-momentum thresholds can be customized. The command-line
tool silently skips events that fail this cleaned-muon multiplicity requirement:

```python
fig, ax, event = display_event(
    source,
    entry=42,
    objects=("Jet", "Muon"),
    min_pt={"Jet": 30, "Muon": 5},
    annotate=False,
)
```

From the command line, the output defaults to `event_<event number>.png`:

```bash
SOURCE="root://cmseos.fnal.gov//store/path/to/file.root"
python -m plotting.event_display "$SOURCE" --entry 42 -o event_42.png
```

An event can instead be selected by its NanoAOD identifiers:

```bash
python -m plotting.event_display "$SOURCE" \
    --run 315257 --lumi 12 --event 123456789
```

The loose and tight validation-region selections from
`workflows/SUEP_coffea_VR.py` can be applied before plotting:

```bash
python -m plotting.event_display "$SOURCE" --entry 42 --VRloose
python -m plotting.event_display "$SOURCE" --entry 57 --VRtight
```

The equivalent notebook argument is `selection="VRloose"` or
`selection="VRtight"`. A selected event displays only the muons retained by
that VR definition; jets, electrons, and photons are unchanged. If the event
does not pass, the tool reports the failed requirement and does not write a
plot. These options reproduce the `apply_VR` region cuts without Rochester
corrections. Dataset-level trigger, trigger-plateau, and golden-JSON filtering
remain upstream, as in the analysis workflow.

Run `python -m plotting.event_display --help` for object filters, individual
minimum-pT settings, eta range, jet radius, labels, and interactive display
options. Remote `root://` input requires the usual XRootD setup and, where
needed, a valid CMS proxy.
