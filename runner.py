import argparse
import importlib
import inspect
import json
import os
import pickle
import sys
from typing import Optional, Union

from coffea import nanoevents, processor
from coffea.processor import Accumulatable
from rich import pretty  # type: ignore[import]

from workflows.SUEP_coffea import SUEP_cluster

# Make this script work from current directory
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)


def loadder(args: argparse.Namespace) -> dict:
    with open(args.samplejson) as f:
        sample_dict = json.load(f)
    for key in sample_dict.keys():
        sample_dict[key] = sample_dict[key][: args.limit]
    if args.executor == "dask/casa":
        for key in sample_dict.keys():
            print(key)
            sample_dict[key] = [
                "root://xcache//" + path.split("//")[-1] for path in sample_dict[key]
            ]
    return sample_dict


def getXSection(
    dataset: str, year: str, SUEP: Optional[bool] = False, path: Optional[str] = "data/"
) -> float:
    filename = f"{path}/xsections_{year}_{'SUEP' if SUEP else ''}.json"

    try:
        with open(filename) as file:
            MC_xsecs = json.load(file)

        if SUEP:
            return MC_xsecs[dataset]

        return (
            MC_xsecs[dataset]["xsec"]
            * MC_xsecs[dataset]["kr"]
            * MC_xsecs[dataset]["br"]
        )

    except (KeyError, FileNotFoundError) as e:
        print(
            f"WARNING: Could not find xsection for {dataset}. Check dataset name and json file."
        )
        return 1


def setup_workflow(
    workflow_name: str, args: argparse.Namespace, sample_dict: dict
) -> SUEP_cluster:
    """
    Dynamically import and setup workflow from the workflow name

    Args:
        workflow_name: Name of the workflow (will be used to construct file/class name)
        args: Command line arguments
        sample_dict: Dictionary of samples to process

    Returns:
        Workflow processor instance
    """
    try:
        # Construct the module name
        module_name = f"workflows.{workflow_name}"

        # Import the module
        module = importlib.import_module(module_name)

        # Get the SUEP_cluster class from the module
        workflow_class = getattr(module, "SUEP_cluster")

        # Default list of all parameters
        params = {
            "isMC": args.isMC,
            "era": args.era,
            "do_syst": args.doSyst,
            "syst_var": "",
            "sample": sample_dict,
            "weight_syst": False,
            "flag": False,
            "output_location": os.getcwd(),
            "accum": args.executor,
            "trigger": args.trigger,
            "debug": args.debug,
            "scouting": args.scouting,
            "do_inf": args.doInf,
            "blind": True,
            "region": args.region,
        }

        # Check if the parameters are valid for the workflow
        params_used = {}
        signature = inspect.signature(workflow_class.__init__)
        for param in params:
            if param in signature.parameters:
                params_used[param] = params[param]

        # Create and return the workflow instance
        return workflow_class(**params_used)

    except ImportError as e:
        raise ImportError(f"Could not import workflow '{workflow_name}'. Error: {e}")
    except AttributeError as e:
        raise AttributeError(
            f"Workflow module '{workflow_name}' must contain a 'SUEP_cluster' class. Error: {e}"
        )


def get_main_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run analysis on baconbits files using processor coffea files"
    )
    # Inputs
    parser.add_argument(
        "-w",
        "--workflow",
        type=str,
        help="Name of the workflow to run (will be used to import SUEP_coffea_<workflow>)",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Prefix for the output file name. The file name will have the form: <prefix>_<sample>.hdf5. (default: %(default)s)",
        required=False,
    )
    parser.add_argument(
        "--samples",
        "--json",
        dest="samplejson",
        default="filelist/SUEP_files_simple.json",
        help="JSON file containing dataset and file locations (default: %(default)s)",
    )

    # Scale out
    parser.add_argument(
        "--executor",
        choices=[
            "iterative",
            "futures",
            "dask/condor",
            "dask/lpc",
            "dask/lxplus",
            "dask/casa",
        ],
        default="futures",
        help="The type of executor to use (default: %(default)s). Other options can be implemented. "
        "For example see https://parsl.readthedocs.io/en/stable/userguide/configuring.html"
        "- `dask/condor` - tested at DESY, RWTH"
        "- `dask/lpc` - custom lpc/condor setup (due to write access restrictions)"
        "- `dask/lxplus` - custom lxplus/condor setup (due to port restrictions)",
    )
    parser.add_argument(
        "-j",
        "--workers",
        type=int,
        default=12,
        help="Number of workers (cores/threads) to use for multi-worker executors "
        "(e.g. futures or condor) (default: %(default)s)",
    )
    parser.add_argument(
        "-s",
        "--scaleout",
        type=int,
        default=1,
        help="Number of nodes to scale out to if using slurm/condor. Total number of "
        "concurrent threads is ``workers x scaleout`` (default: %(default)s)",
    )
    parser.add_argument(
        "--max-scaleout",
        dest="max_scaleout",
        type=int,
        default=250,
        help="The maximum number of nodes to adapt the cluster to. (default: %(default)s)",
    )
    parser.add_argument(
        "--voms",
        default=None,
        type=str,
        help="Path to voms proxy, accessible to worker nodes. By default a copy will be made to $HOME.",
    )
    # Debugging
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Do not process, just check all files are accessible",
    )
    parser.add_argument("--skipbadfiles", action="store_true", help="Skip bad files.")
    parser.add_argument(
        "--only", type=str, default=None, help="Only process specific dataset or file"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Limit to the first N files of each dataset in sample JSON",
    )
    parser.add_argument(
        "--chunk",
        type=int,
        default=15000,
        metavar="N",
        help="Number of events per process chunk",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=None,
        metavar="N",
        help="Max number of chunks to run in total",
    )
    parser.add_argument(
        "--mild_scaleout",
        action="store_true",
        help="Parameters for mild scaleout. Use when the scheduler is empty.",
    )
    parser.add_argument(
        "--memory", type=str, default="2GB", help="Change worker memory"
    )
    parser.add_argument(
        "--isMC", action="store_true", help="Specify if the file is MC or data"
    )
    parser.add_argument("--era", type=str, default="2018", help="Specify the year")
    parser.add_argument(
        "--doSyst", action="store_true", help="Turn systematics on or off"
    )
    parser.add_argument(
        "--scouting", action="store_true", help="Turn processing for scouting on"
    )
    parser.add_argument("--doInf", action="store_true", help="Turn inference on")
    parser.add_argument("--dataset", type=str, help="Dataset to find xsection")
    parser.add_argument(
        "--trigger", type=str, default="PFHT", help="Specify HLT trigger path"
    )
    parser.add_argument("--skimmed", action="store_true", help="Use skimmed files")
    parser.add_argument(
        "--region", type=str, default="", help="Specify the region to run on"
    )
    parser.add_argument("--debug", action="store_true", help="Turn debugging on")
    parser.add_argument("--verbose", action="store_true", help="Turn verbose on")
    parser.add_argument("--check_hlt", action="store_true", help="Check HLT paths")
    return parser


def specificProcessing(args: argparse.Namespace, sample_dict: dict) -> dict:
    if args.only in sample_dict.keys():  # is dataset
        sample_dict = dict([(args.only, sample_dict[args.only])])
    if "*" in args.only:  # wildcard for datasets
        _new_dict = {}
        print("Will only process the following datasets:")
        for k, v in sample_dict.items():
            if k.lstrip("/").startswith(args.only.rstrip("*")):
                print("    ", k)
                _new_dict[k] = v
        sample_dict = _new_dict
    else:  # is file
        for key in sample_dict.keys():
            if args.only in sample_dict[key]:
                sample_dict = dict([(key, [args.only])])
    return sample_dict


def daskExecutor(args: argparse.Namespace) -> processor.DaskExecutor:
    import shutil

    from dask.distributed import Client, Worker, WorkerPlugin  # type: ignore[import]
    from distributed.diagnostics.plugin import UploadDirectory  # type: ignore[import]

    # Define the class once, outside the conditional blocks
    class SettingSitePath(WorkerPlugin):
        def __init__(self, base_path: str):
            self.base_path = base_path

        def setup(self, worker: Worker):
            sys.path.insert(0, os.getcwd() + self.base_path)

    if "lpc" in args.executor:
        from lpcjobqueue import LPCCondorCluster  # type: ignore[import]

        cluster = LPCCondorCluster(
            transfer_input_files="/srv/workflows/",
            shared_temp_directory="/tmp",
            memory=args.memory,
            worker_extra_args=[
                "--worker-port 10000:10070",
                "--nanny-port 10070:10100",
                "--no-dashboard",
            ],
            job_script_prologue=[],
            log_directory="/uscmst1b_scratch/lpc1/3DayLifetime/chpapage/",
            scheduler_options={"dashboard_address": ":44890"},
        )
        adapt_parameters = {"wait_count": 10}
        if args.mild_scaleout:
            adapt_parameters = dict(
                interval="1m",
                target_duration="30s",
                wait_count=10,
            )
        cluster.adapt(
            minimum=args.scaleout,
            maximum=args.max_scaleout,
            **adapt_parameters,
        )
        client = Client(cluster)

        client.register_plugin(UploadDirectory(os.getcwd() + "/data"))
        client.register_plugin(SettingSitePath("/workflows/"))
        shutil.make_archive("workflows", "zip", base_dir="workflows")
        client.upload_file("workflows.zip")

        print("Waiting for at least one worker...")
        client.wait_for_workers(1)

    elif "casa" in args.executor:
        client = Client("tls://localhost:8786")
        client.register_plugin(UploadDirectory(os.getcwd() + "/data"))
        client.register_plugin(SettingSitePath("/dask-worker-space/"))
        shutil.make_archive("workflows", "zip", base_dir="workflows")
        client.upload_file("workflows.zip")
    else:
        raise NotImplementedError(f"I don't know anything about {args.executor}.")

    return processor.DaskExecutor(client=client)


def nativeExecutors(
    args: argparse.Namespace,
) -> Union[processor.IterativeExecutor, processor.FuturesExecutor]:
    executor = processor.IterativeExecutor()
    if args.executor == "futures":
        executor = processor.FuturesExecutor(workers=args.workers)
    return executor


def getWeights(sample_dict: dict) -> Accumulatable:
    from workflows.GenSumWeightExtract import GenSumWeightExtractor

    genSumW_instance = GenSumWeightExtractor()
    genSumW_executor = processor.IterativeExecutor()
    genSumW_run = processor.Runner(
        executor=genSumW_executor,
        schema=nanoevents.BaseSchema,  # type: ignore[import]
        align_clusters=True,
    )
    genSumW = genSumW_run(
        fileset=sample_dict,
        treename="Runs",
        processor_instance=genSumW_instance,
    )
    return genSumW


def checkHLTpaths(sample_dict: dict) -> Accumulatable:
    from workflows.CheckHLTpaths import CheckHLTpaths

    hlt_instance = CheckHLTpaths()
    hlt_executor = processor.FuturesExecutor(workers=args.workers)
    hlt_run = processor.Runner(
        executor=hlt_executor,
        schema=nanoevents.NanoAODSchema,  # type: ignore[import]
        align_clusters=True,
    )
    hlt = hlt_run(
        fileset=sample_dict,
        treename="Events",
        processor_instance=hlt_instance,
    )
    return hlt


def execute(
    args: argparse.Namespace, processor_instance: SUEP_cluster, sample_dict: dict
) -> Accumulatable:
    """
    Main function to execute the workflow
    """
    if args.executor in ["futures", "iterative"]:
        executor = nativeExecutors(args)
    elif "dask" in args.executor:
        executor = daskExecutor(args)
    else:
        raise NotImplementedError

    run = processor.Runner(
        executor=executor,
        chunksize=args.chunk,
        maxchunks=args.max,
        schema=nanoevents.NanoAODSchema,  # type: ignore[import]
        skipbadfiles=args.skipbadfiles,
    )
    output = run(
        fileset=sample_dict,
        treename="Events",
        processor_instance=processor_instance,
    )

    return output


def saveOutput(
    args: argparse.Namespace,
    output: dict,
    sample: str,
    gensumweight: Optional[float] = None,
) -> None:
    """
    Save the output to file(s)
    Will calculate weights if necessary
    """

    if gensumweight is not None:
        output["gensumweight"].value = gensumweight
        output["cutflow"][0] = [gensumweight, gensumweight]

    if args.isMC:
        xsection = getXSection(sample, args.era)
        scale = xsection / output["gensumweight"].value
        pretty.pprint(
            f"Scaling {sample} by {xsection} / {output['gensumweight'].value} = {scale}"
        )

    # Save the output
    outputName = ""
    if args.output is not None:
        outputName = f"{args.output}_"
    outputName = f"{outputName}{sample}"

    # Save the cutflow (normalized to the gensumweight)
    if "cutflow" in output.keys():
        cutflowName = f"{outputName}_cutflow.pkl"
        if args.isMC:
            output["cutflow"] *= scale
        print(f"Saving the following cutflow to {cutflowName}")
        pickle.dump(output["cutflow"], open(cutflowName, "wb"))

    # Save the histograms (normalized to the gensumweight)
    if "histograms" in output.keys():
        histName = f"{outputName}_histograms.pkl"
        if args.isMC:
            for p in output["histograms"].keys():
                output["histograms"][p] *= scale
        print(f"Saving the following histograms to {histName}")
        pickle.dump(output["histograms"], open(histName, "wb"))


if __name__ == "__main__":
    parser = get_main_parser()
    args = parser.parse_args()

    # Load dataset
    sample_dict = loadder(args)

    # For debugging
    if args.only:
        sample_dict = specificProcessing(args, sample_dict)

    # Check HLT paths
    if args.check_hlt:
        hlt = checkHLTpaths(sample_dict)
        print(hlt)
        sys.exit(0)

    # Load workflow using dynamic import
    processor_instance = setup_workflow(args.workflow, args, sample_dict)

    # Execute the workflow
    output = execute(args, processor_instance, sample_dict)

    # Calculate the gen sum weight for skimmed samples
    if args.skimmed:
        weights = getWeights(sample_dict)
        print(
            "You are using skimmed data! I was able to retrieve the following gensum weights:\n"
        )
        pretty.pprint(weights)

    # Save the output
    for sample in sample_dict:
        if args.skimmed:
            weight = weights[sample]  # type: ignore[import]
            if not isinstance(weight, int):
                weight = weight.value
            saveOutput(args, output[sample], sample, gensumweight=weight)  # type: ignore[import]
        else:
            saveOutput(args, output[sample], sample)  # type: ignore[import]

    if args.verbose:
        pretty.pprint(output)
