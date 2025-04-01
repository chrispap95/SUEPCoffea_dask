import logging
import math
import os
import pickle
import re
from pathlib import Path
from typing import Optional

import dataset_groups
import hist
import matplotlib.gridspec as gridspec  # type: ignore[import]
import matplotlib.pyplot as plt  # type: ignore[import]
import matplotlib.ticker as ticker  # type: ignore[import]
import mplhep as hep
import numpy as np
import ROOT  # type: ignore[import]
import scipy.special as special  # type: ignore[import]
import uproot
from iminuit import Minuit
from iminuit.cost import LeastSquares
from jacobi import propagate  # type: ignore[import]
from rich.pretty import pprint  # type: ignore[import]
from rich.progress import track  # type: ignore[import]

# https://twiki.cern.ch/twiki/bin/viewauth/CMS/RA2b13TeVProduction#Dataset_luminosities_2016_pb_1
lumis = {
    "2016_apv": 19497.914,
    "2016": 16810.813,
    "2017": 41471.589,
    # NOTE: Only 2018 lumi has been properly calculated
    "2018": 59795.400422,
}


def lumi_Label(year: str) -> float:
    """
    Return the luminosity in /fb for a given year.
    """
    if year == "2016":
        return round((lumis[year] + lumis[year + "_apv"]) / 1000, 1)
    return round(lumis[year] / 1000, 1)


def find_lumi(
    infile_name: str,
    auto_lumi: bool = True,
    year: Optional[int | str] = None,
) -> float:
    """
    Find the luminosity for a given input file.

    Parameters
    ----------
    infile_name : str
        The name of the input file.
    auto_lumi : bool
        Automatically determine the luminosity based on the input file name.
    year : int or str
        The year to use for the luminosity. If not provided, use the auto_lumi option.

    Returns
    -------
    float
        The luminosity in /pb.
    """
    if isinstance(year, int):
        year = str(year)

    if auto_lumi:
        # NOTE: auto_lumi is broken for Nov2024 UL samples because of naming
        if "20UL16MiniAODv2" in infile_name:
            lumi = lumis["2016"]
        if "20UL17MiniAODv2" in infile_name:
            lumi = lumis["2017"]
        if "20UL16MiniAODAPVv2" in infile_name:
            lumi = lumis["2016_apv"]
        if "20UL18" in infile_name:
            lumi = lumis["2018"]
        if "SUEP" in infile_name:
            lumi = lumis["2018"]
        if "DoubleMuon" in infile_name:
            lumi = 1
    if year and not auto_lumi:
        lumi = lumis[year]
    if year and auto_lumi:
        raise Exception("Apply lumis automatically or based on year")
    return lumi


def load_samples(
    infile_names: list[str],
    year: Optional[int | str] = None,
    auto_lumi: bool = True,
    custom_lumi: Optional[float] = None,
    is_data: bool = False,
) -> dict:
    """
    Load histograms from a list of input files.

    Parameters
    ----------
    infile_names : list
        List of input file names.
    year : int or str
        The year to use for the luminosity.
    auto_lumi : bool
        Automatically determine the luminosity based on the input file name.
    custom_lumi : float
        Custom luminosity in /pb.
    is_data : bool
        Flag to indicate if the input is data.

    Returns
    -------
    dict
        Dictionary of histograms.
    """
    plots = {}
    # Load histograms and scale to lumi
    for infile_name in infile_names:
        if not os.path.isfile(infile_name):
            print("WARNING:", infile_name, "doesn't exist")
            continue
        elif ".pkl" not in infile_name:
            print("WARNING:", infile_name, "is not a .pkl file")
            continue

        # Set the lumi based on year or override using a custom value (in /pb).
        # Data shouldn't be scaled.
        if year is not None:
            auto_lumi = False
        lumi = find_lumi(
            infile_name,
            auto_lumi,
            year,
        )
        if custom_lumi is not None:
            lumi = custom_lumi
        if is_data:
            lumi = 1

        sample = infile_name.split("/")[-1].replace("_histograms.pkl", "")

        with open(infile_name, "rb") as f:
            plots[sample] = pickle.load(f)
        for plot in plots[sample]:
            plots[sample][plot] = plots[sample][plot] * lumi

    # Create combined histograms
    for combined_dataset in dataset_groups.dataset_groups_new:
        temp_dict = {}
        for pattern in dataset_groups.dataset_groups_new[combined_dataset]:
            for sample in plots:
                if re.search(pattern, sample):
                    for plot in plots[sample]:
                        if plot not in temp_dict:
                            temp_dict[plot] = plots[sample][plot].copy()
                        else:
                            temp_dict[plot] += plots[sample][plot]
        if len(temp_dict) > 0:
            plots[combined_dataset] = temp_dict.copy()
    return plots


def loader(
    tag: str,
    custom_lumi: Optional[float] = None,
    load_data: bool = False,
    input_dir: Optional[str] = None,
    verbosity: int = 0,
) -> dict:
    """
    Load histograms from the processor output files.

    Parameters
    ----------
    tag : str
        The tag used to identify the processor output files.
        Will get the files from the directory: ../../processor_output_files/{tag}_output_histograms/
    custom_lumi : float
        Will use this luminosity (in /pb) if provided.
    load_data : bool
        Flag to indicate if data should be loaded.
    input_dir : str
        Input directory for the processor output files. it will override the default directory.
    verbosity : int
        Verbosity level.

    Returns
    -------
    dict
        Dictionary of histograms.
    """
    # input .pkl files
    base_dir = Path(__file__).parent.parent
    if input_dir:
        base_dir = Path(input_dir)
    plot_dir = base_dir / "processor_output_files" / f"{tag}_output_histograms/"
    if verbosity > 0:
        print(f"Loading histograms from {plot_dir}")
    filenames = list(plot_dir.glob("*histograms.pkl"))
    basenames = [str(f.name) for f in filenames]

    # separate the files into signal, background, and data
    files_SUEP = [
        str(plot_dir / b) for b in basenames if ("SUEP" in b) or ("ggHBSMpythia" in b)
    ]
    files_bkg = [
        str(plot_dir / b)
        for b in basenames
        if ("pythia8" in b) and ("SUEP" not in b) and ("ggHBSMpythia" not in b)
    ]
    files_data = [str(plot_dir / b) for b in basenames if ("DoubleMuon" in b)]

    if verbosity > 0:
        pprint(files_bkg)

    # load histograms and scale to lumi
    plots_SUEP_2018 = load_samples(files_SUEP, year=2018, custom_lumi=custom_lumi)
    plots_bkg_2018 = load_samples(files_bkg, year=2018, custom_lumi=custom_lumi)
    if load_data:
        plots_data_2018 = load_samples(files_data, year=2018, is_data=True)

    if verbosity > 1:
        pprint(plots_SUEP_2018)

    # put everything in one dictionary
    plots = {}
    for sample in plots_SUEP_2018:
        if sample + "_2018" not in plots:
            plots[sample + "_2018"] = plots_SUEP_2018[sample]
        else:
            plots[sample + "_2018"].update(plots_SUEP_2018[sample])
    for sample in plots_bkg_2018:
        if sample + "_2018" not in plots:
            plots[sample + "_2018"] = plots_bkg_2018[sample]
        else:
            plots[sample + "_2018"].update(plots_bkg_2018[sample])
    if load_data:
        for sample in plots_data_2018:
            if sample + "_2018" not in plots:
                plots[sample + "_2018"] = plots_data_2018[sample]
            else:
                plots[sample + "_2018"].update(plots_data_2018[sample])

    return plots


class Extrapolation:
    """
    Class to perform extrapolation of histograms.
    """

    def __init__(
        self,
        plots: dict,
        is_data: bool = False,
        fit_function: str = "exponential",
        uncertainty_scheme: str = "simple",
    ) -> None:
        """
        Initialize the class.

        Parameters
        ----------
        plots : dict
            Dictionary of histograms.
        is_data : bool
            Flag to indicate if the input is data.
        fit_function : str
            The fit function to use. Options are "exponential" or "binomial".
            Exponential is a simple power law fit.
            Binomial is a more complex function that uses the binomial distribution.
            Default is "exponential".
        uncertainty_scheme : str
            The uncertainty scheme to use. Options are "simple" or "full".
            Simple uses the uncertainty from the fit.
            Full adds the MC statistical uncertainty and the uncertainty from the fit in quadrature.
        """
        self.plots = plots
        self.fit_results = {}
        self.is_data = is_data
        self.fit_function = fit_function
        self.uncertainty_scheme = uncertainty_scheme

    def muon_func_log(self, x: np.ndarray, loga: float, logb: float) -> np.ndarray:
        """
        This function is the log of a power law function: a * b**-n
        """
        return loga - x * logb

    def muon_func_tight_log(
        self, x: np.ndarray, loga_t: float, logb: float
    ) -> np.ndarray:
        """
        Wrapper for muon_func_log. To be used in the fit for the tight region
        """
        return self.muon_func_log(x, loga_t, logb)

    def muon_func_loose_log(
        self, x: np.ndarray, loga_l: float, logb: float
    ) -> np.ndarray:
        """
        Wrapper for muon_func_log. To be used in the fit for the loose region
        """
        return self.muon_func_log(x, loga_l, logb)

    def muon_func_log_alt(
        self, x: np.ndarray, loga: float, logb: float, N: float
    ) -> np.ndarray:
        """
        This function is the log of effective binomial distribution: a * b**n * (1-b)**(N-n)

        Binomial factor using gamma function:
        N!/n!(N-n)! = gamma(N+1)/(gamma(n+1)*gamma(N-n+1))
        """
        binomial_factor = special.gamma(N + 1) / (
            special.gamma(x + 1) * special.gamma(N - x + 1)
        )
        return (
            loga
            + np.log(binomial_factor)
            + x * logb
            + (N - x) * np.log(1 - np.exp(logb))
        )

    def muon_func_tight_log_alt(
        self, x: np.ndarray, loga_t: float, logb: float, N: float
    ) -> np.ndarray:
        """
        Wrapper for muon_func_log_alt. To be used in the fit for the tight region
        """
        return self.muon_func_log_alt(x, loga_t, logb, N)

    def muon_func_loose_log_alt(
        self, x: np.ndarray, loga_l: float, logb: float, N: float
    ) -> np.ndarray:
        """
        Wrapper for muon_func_log_alt. To be used in the fit for the loose region
        """
        return self.muon_func_log_alt(x, loga_l, logb, N)

    def extrapolate(
        self, slice_hists: dict = {}, syst: str = "", verbose: bool = False
    ) -> None:
        """
        Perform extrapolation of histograms.

        Parameters
        ----------
        slice_hists : dict
            Dictionary of histogram keys and slice() objects that should be applied to
            them if provided.
        syst : str
            The systematic variation to use.
        verbose : bool
            Flag to indicate if the fit results should be printed.
        """
        if not syst.startswith("_") and syst != "":
            syst = f"_{syst}"
        for region in self.find_extrapolatable_regions(syst):
            # Perform and evaluate the fit
            m = self.fit(region, slice_hists=slice_hists, syst=syst, verbose=verbose)
            h_loose = self.evaluate_fit(
                f"{region}_loose", m, slice_hists=slice_hists, syst=syst
            )
            h_tight = self.evaluate_fit(
                f"{region}_tight", m, slice_hists=slice_hists, syst=syst
            )

            # Store the fit results and extrapolated histograms
            self.fit_results[f"{region}{syst}"] = m
            self.plots[f"{region}_loose_extrapolation{syst}"] = h_loose
            self.plots[f"{region}_tight_extrapolation{syst}"] = h_tight

    def find_syst_variations(self) -> None:
        """
        Find all systematic variations by looking at all available plots.
        """
        syst_variations = set()
        for region in self.plots:
            if "_tight" in region:
                syst_variations.add(
                    region.split("_tight")[-1]
                    .replace("extrapolation", "")
                    .replace("_", "")
                )
        self.syst_variations = list(syst_variations)
        return

    def get_syst_variations(self) -> list[str]:
        if not hasattr(self, "syst_variations"):
            self.find_syst_variations()
        return self.syst_variations

    def fit_syst_variations(self, slice_hists: dict, verbose: bool):
        """
        Perform the extrapolation for all systematic variations of a region.
        """
        for syst_var in self.get_syst_variations():
            if verbose and syst_var != "":
                print(f"Extrapolating {syst_var}")
            elif verbose:
                print("Extrapolating nominal")
            self.extrapolate(slice_hists=slice_hists, syst=syst_var, verbose=verbose)

    def find_extrapolatable_regions(self, syst: str) -> list[str]:
        """
        Find all regions that can be extrapolated.
        """
        return [
            region.replace("_tight", "").replace("_extrapolation", "").replace(syst, "")
            for region in self.plots
            if region.endswith(f"tight{syst}")
        ]

    def sanitize_hist(
        self, h: hist.Hist | hist.accumulators.WeightedSum
    ) -> hist.Hist | hist.accumulators.WeightedSum:
        """
        Remove bins with zero content from histogram.
        """
        if isinstance(h, hist.Hist):
            zero_bins = np.where(h.values() == 0)[0]
            if len(zero_bins) == 0:
                return h
            if (zero_bins[-1] - zero_bins[0]) != (len(zero_bins) - 1):
                print("Warning: zero bins are not contiguous")
            return h[: int(zero_bins[0])]
        elif isinstance(h, hist.accumulators.WeightedSum):
            if h.value == 0:
                raise ValueError("Cannot fit a histogram with zero content")
            return h
        else:
            raise TypeError(f"{h} must be a hist.Hist or hist.accumulators.WeightedSum")

    def get_values(self, h: hist.Hist | hist.accumulators.WeightedSum) -> np.ndarray:
        if isinstance(h, hist.Hist):
            return h.values()
        elif isinstance(h, hist.accumulators.WeightedSum):
            return np.array([h.value])
        else:
            raise TypeError(f"{h} must be a hist.Hist or hist.accumulators.WeightedSum")

    def get_variances(self, h: hist.Hist | hist.accumulators.WeightedSum) -> np.ndarray:
        if isinstance(h, hist.Hist):
            return h.variances()
        elif isinstance(h, hist.accumulators.WeightedSum):
            return np.array([h.variance])
        else:
            raise TypeError(f"{h} must be a hist.Hist or hist.accumulators.WeightedSum")

    def fit(self, region: str, slice_hists: dict, syst: str, verbose: bool) -> Minuit:
        """
        Perform simultaneous Least Squares fit to the loose and tight regions.

        Parameters
        ----------
        region : str
            The region to fit. E.g. "SR_high_temp".
        slice_hists : dict
            Dictionary of histogram keys and slice() objects that should be applied to
            them if provided.
        syst : str
            The systematic variation to use.
        verbose : bool
            Flag to indicate if the fit results should be printed.

        Returns
        -------
        Minuit
            The Minuit fit result.
        """
        if f"{region}_loose{syst}" in slice_hists:
            slc_l = slice_hists[f"{region}_loose{syst}"]
        elif f"{region}_loose" in slice_hists:
            logging.warning(f"Using {region}_loose slice for {region}_loose{syst}")
            slc_l = slice_hists[f"{region}_loose"]
        else:
            slc_l = slice(None)
        if f"{region}_tight{syst}" in slice_hists:
            slc_t = slice_hists[f"{region}_tight{syst}"]
        elif f"{region}_tight" in slice_hists:
            logging.warning(f"Using {region}_tight slice for {region}_tight{syst}")
            slc_t = slice_hists[f"{region}_tight"]
        else:
            slc_t = slice(None)

        h_l = self.plots[f"{region}_loose{syst}"][slc_l]
        h_t = self.plots[f"{region}_tight{syst}"][slc_t]

        h_l = self.sanitize_hist(h_l)
        h_t = self.sanitize_hist(h_t)

        # Convert histograms to log10 arrays.
        data_y_l = np.log10(self.get_values(h_l))
        data_y_t = np.log10(self.get_values(h_t))
        data_yerr_l = np.sqrt(self.get_variances(h_l)) / (
            self.get_values(h_l) * np.log(10)
        )
        data_yerr_t = np.sqrt(self.get_variances(h_t)) / (
            self.get_values(h_t) * np.log(10)
        )
        data_x_l = np.arange(len(data_y_l))
        data_x_t = np.arange(len(data_y_t))

        if self.fit_function == "exponential":
            fit_function_tight = self.muon_func_tight_log
            fit_function_loose = self.muon_func_loose_log
            parameters = {"loga_t": 1, "loga_l": 1, "logb": 1}
        elif self.fit_function == "binomial":
            fit_function_tight = self.muon_func_tight_log_alt
            fit_function_loose = self.muon_func_loose_log_alt
            parameters = {"loga_t": 7, "loga_l": 8, "logb": -2.3, "N": 7}
        else:
            raise ValueError("Invalid fit function")

        least_squares = LeastSquares(
            data_x_l, data_y_l, data_yerr_l, fit_function_loose  # type: ignore[arg-type]
        ) + LeastSquares(
            data_x_t, data_y_t, data_yerr_t, fit_function_tight  # type: ignore[arg-type]
        )
        m = Minuit(least_squares, **parameters)  # type: ignore[arg-type]
        m.migrad()
        m.hesse()

        if verbose:
            print(f"Fit results for {region}{syst}:")
            print(m)

        return m

    def print_fit_results(self) -> None:
        """
        Print the fit results for all regions.
        """
        for region in self.fit_results:
            print(f"Fit results for {region}:")
            print(self.fit_results[region])

    def evaluate_fit(
        self, region: str, m: Minuit, slice_hists: dict, syst: str
    ) -> hist.Hist:
        """
        This function accepts a Minuit fit result.
        It will output a histogram that represents an evaluation of the fit.

        Parameters
        ----------
        region : str
            The region to evaluate. E.g. "SR_high_temp".
        m : Minuit
            The Minuit fit result.
        slice_hists : dict
            Dictionary of histogram keys and slice() objects that were be applied to
            the histograms when fitting. Needed to determine the x value to begin the
            fit evaluation.
        syst : str
            The systematic variation to use.

        Returns
        -------
        hist.Hist
            The evaluated histogram.
        """
        h0 = self.plots[region].copy().reset()
        x = h0.axes[0].edges[:-1]

        # Extract the fit parameters and covariance matrix
        loga = "loga_t" if "tight" in region else "loga_l"
        if self.fit_function == "exponential":
            params = np.array([m.values[loga], m.values["logb"]])
            if m.covariance is not None:
                cov = np.array(
                    [
                        [m.covariance[loga, loga], m.covariance[loga, "logb"]],
                        [m.covariance["logb", loga], m.covariance["logb", "logb"]],
                    ]
                )
            else:
                raise ValueError("Covariance matrix is None")
        elif self.fit_function == "binomial":
            params = np.array([m.values[loga], m.values["logb"], m.values["N"]])
            if m.covariance is not None:
                cov = np.array(
                    [
                        [
                            m.covariance[loga, loga],
                            m.covariance[loga, "logb"],
                            m.covariance[loga, "N"],
                        ],
                        [
                            m.covariance["logb", loga],
                            m.covariance["logb", "logb"],
                            m.covariance["logb", "N"],
                        ],
                        [
                            m.covariance["N", loga],
                            m.covariance["N", "logb"],
                            m.covariance["N", "N"],
                        ],
                    ]
                )
            else:
                raise ValueError("Covariance matrix is None")
        else:
            raise ValueError("Invalid fit function")

        # Determine the x value to begin the fit
        fit_x_begin = x[0]
        if f"{region}_{syst}" in slice_hists:
            fit_x_begin = slice_hists[f"{region}_{syst}"].start.imag
        elif region in slice_hists:
            fit_x_begin = slice_hists[region].start.imag

        if self.fit_function == "exponential":
            fit_function = self.muon_func_log
        elif self.fit_function == "binomial":
            fit_function = self.muon_func_log_alt
        else:
            raise ValueError("Invalid fit function")

        # Propagate the fit parameters and covariance matrix
        logy, logycov = propagate(
            lambda p: fit_function(x - fit_x_begin, *p), params, cov
        )
        logyerr_prop = np.diag(logycov) ** 0.5
        y = 10**logy
        yerr_prop = np.log(10) * y * logyerr_prop

        # Fill the histogram and return it
        if self.uncertainty_scheme == "simple":
            for i in range(len(h0.values())):
                h0[i] = (y[i], yerr_prop[i] ** 2)
        elif self.uncertainty_scheme == "full":
            mc_vals = self.plots[region].values()
            mc_vars = self.plots[region].variances()
            mc_rel_unc = np.divide(
                np.sqrt(mc_vars),
                mc_vals,
                where=mc_vals != 0,
                out=np.zeros_like(mc_vals),
            )
            # Find closest non-zero values
            non_zero_indices = np.where(mc_rel_unc != 0)[0]
            indices = np.indices(mc_rel_unc.shape)[0]
            closest_indices = non_zero_indices[
                np.argmin(np.abs(indices[:, None] - non_zero_indices), axis=1)
            ]
            mc_rel_unc = mc_rel_unc[closest_indices]
            for i in range(len(h0.values())):
                h0[i] = (y[i], (mc_rel_unc[i] * y[i]) ** 2 + yerr_prop[i] ** 2)
        else:
            raise ValueError("Invalid uncertainty scheme")

        return h0

    def plot_region(
        self, region: str, syst: str, ax1: plt.Axes, ax2: plt.Axes, ax3: plt.Axes
    ) -> None:
        """
        Function to be called by plot_fit. Plots the region's MC histograms and extrapolations,
        as well as the ratio and pulls.

        Parameters
        ----------
        region : str
            The region to plot. E.g. "SR_high_temp".
        syst : str
            The systematic variation to use. E.g. "TrkEff". Default is no variation.
        ax1 : plt.Axes
            The axes to plot the histograms on.
        ax2 : plt.Axes
            The axes to plot the ratio on.
        ax3 : plt.Axes
            The axes to plot the pulls on.
        """
        plot_pre_fit = self.plots[f"{region}"]
        post_fit_name = (
            f"{region}_extrapolation"
            if syst == ""
            else f"{region.replace(syst, '')}_extrapolation{syst}"
        )
        plot_post_fit = self.plots[post_fit_name]

        y = plot_post_fit.values()
        yerr_prop = np.sqrt(plot_post_fit.variances())

        x_hatch = np.vstack(
            (plot_pre_fit.axes[0].edges[:-1], plot_pre_fit.axes[0].edges[1:])
        ).reshape((-1,), order="F")
        y_hatch = np.vstack((y, y)).reshape((-1,), order="F")
        y_hatch_unc = np.vstack((yerr_prop, yerr_prop)).reshape((-1,), order="F")
        hep.histplot(
            plot_pre_fit,
            yerr=np.sqrt(plot_pre_fit.variances()),
            label="data" if self.is_data else "MC",
            ax=ax1,
        )
        hep.histplot(y, bins=plot_post_fit.axes[0].edges, label="fit", ax=ax1)
        ax1.fill_between(
            x=x_hatch,
            y1=y_hatch - y_hatch_unc,  # type: ignore[arg-type]
            y2=y_hatch + y_hatch_unc,  # type: ignore[arg-type]
            label="fit + stat. unc.",
            step="pre",
            facecolor="C1",
            alpha=0.3,
            linewidth=0,
        )
        ax1.set_title(region.replace("_", " ").replace("temp", "T"))
        ax1.legend()
        ax1.set_yscale("log")
        ax1.set_xlabel("")
        ax1.set_ylabel("Events")
        ax1.set_ylim(1e-4, 1e8)
        ax1.set_xlim(2.8, 8.2)
        ax1.xaxis.set_minor_locator(ticker.NullLocator())
        ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        ratio = plot_pre_fit.values() / y
        ratio_err = np.sqrt(
            (y**-2) * (plot_pre_fit.variances())
            + (plot_pre_fit.values() ** 2 * y**-4) * (yerr_prop**2)
        )
        ax2.errorbar(
            plot_pre_fit.axes[0].centers,
            ratio,
            yerr=ratio_err,
            color="black",
            fmt="o",
            linestyle="none",
        )
        ax2.axhline(1, ls="--", color="gray")
        ax2.set_xlabel("")
        ax2.set_ylabel("data / fit" if self.is_data else "MC / fit")
        ax2.set_xlim(2.8, 8.2)
        ax2.set_ylim(0, 2)

        pulls = (plot_pre_fit.values() - y) / np.sqrt(
            plot_pre_fit.variances() + yerr_prop**2
        )
        pulls_up = np.where(pulls >= 0, pulls, 0)
        pulls_down = np.where(pulls < 0, pulls, 0)

        x_hatch = np.vstack(
            (plot_pre_fit.axes[0].edges[:-1], plot_pre_fit.axes[0].edges[1:])
        ).reshape((-1,), order="F")
        y_hatch_up = np.vstack((pulls_up, pulls_up)).reshape((-1,), order="F")
        y_hatch_down = np.vstack((pulls_down, pulls_down)).reshape((-1,), order="F")

        ax3.fill_between(
            x=x_hatch,
            y1=0,
            y2=y_hatch_up,  # type: ignore[arg-type]
            step="pre",
            facecolor="None",
            edgecolor="red",
            alpha=1,
            linewidth=0,
            hatch="///",
        )
        ax3.fill_between(
            x=x_hatch,
            y1=0,
            y2=y_hatch_down,  # type: ignore[arg-type]
            step="pre",
            facecolor="None",
            edgecolor="blue",
            alpha=1,
            linewidth=0,
            hatch="///",
        )

        ax3.set_xlabel("nMuon")
        ax3.set_ylabel("pull")
        ax3.set_xlim(2.8, 8.2)
        ax3.set_ylim(-2.5, 2.5)

        for label in ax1.xaxis.get_ticklabels():
            label.set_visible(False)
        for label in ax2.xaxis.get_ticklabels():
            label.set_visible(False)

    def plot_fit(
        self, region: str, syst: str = "", add_label: bool = False, add_text: str = ""
    ) -> None:
        """
        Plot the region's MC histograms and extrapolations, as well as the ratio and pulls.
        Will plot both the loose and tight regions side by side.

        Parameters
        ----------
        region : str
            The region to plot. E.g. "SR_high_temp".
        syst : str
            The systematic variation to use. E.g. "TrkEff". Default is no variation.
        add_label : bool
            Flag to indicate if the CMS label & lumi should be added to the plot.
        add_text : str
            Text to add to the plot. Useful for adding the MC sample name or data label.
        """
        if not syst.startswith("_") and syst != "":
            syst = f"_{syst}"

        region = (
            region.replace("_tight", "")
            .replace("_loose", "")
            .replace("_extrapolation", "")
            .replace(syst, "")
        )

        # Find max and min y values for all regions
        max_y = 1
        min_y = 1e8
        for subregion in ["_loose", "_tight"]:
            for suffix in ["", "_extrapolation"]:
                values = self.plots[f"{region}{subregion}{suffix}{syst}"].values()
                if len(values) == 0:
                    continue
                values = values[values > 0]
                max_y = max(max_y, values.max())
                min_y = min(min_y, values.min())

        # Create figure
        fig = plt.figure(figsize=(24, 12))

        # Create two GridSpec layouts, one for left and one for right
        # Each has 5 rows and 1 column
        gs_left = gridspec.GridSpec(5, 1, left=0.08, right=0.47, bottom=0.15)
        gs_right = gridspec.GridSpec(5, 1, left=0.53, right=0.92, bottom=0.15)

        # Create left subplots
        ax1_left = plt.subplot(gs_left[0:3, 0])  # Top 3 rows
        ax2_left = plt.subplot(gs_left[3, 0], sharex=ax1_left)  # 4th row
        ax3_left = plt.subplot(gs_left[4, 0], sharex=ax1_left)  # Bottom row

        # Create right subplots
        ax1_right = plt.subplot(gs_right[0:3, 0])  # Top 3 rows
        ax2_right = plt.subplot(gs_right[3, 0], sharex=ax1_right)  # 4th row
        ax3_right = plt.subplot(gs_right[4, 0], sharex=ax1_right)  # Bottom row

        self.plot_region(f"{region}_loose{syst}", syst, ax1_left, ax2_left, ax3_left)
        self.plot_region(f"{region}_tight{syst}", syst, ax1_right, ax2_right, ax3_right)

        if add_label:
            hep.cms.label(llabel="Preliminary", data=True, lumi=59.8, ax=ax1_left)
            hep.cms.label(llabel="Preliminary", data=True, lumi=59.8, ax=ax1_right)

        if add_text != "":
            ax1_left.text(
                0.5,
                0.9,
                add_text,
                transform=ax1_left.transAxes,
                fontsize=32,
                verticalalignment="top",
                horizontalalignment="center",
            )

        # Add fit results to the left plot
        fit_result = self.fit_results[f"{region}{syst}"]
        fit_rslt_str = "Fit result:\n"
        fit_rslt_str += r"$\chi^2$/ndf = "
        fit_rslt_str += f"{fit_result.fmin.reduced_chi2:.2f}\n"
        if self.fit_function == "exponential":
            fit_rslt_str += r"$A_\text{tight} = $"
            fit_rslt_str += f"{fit_result.values['loga_t']:.3f} ± "
            fit_rslt_str += f"{fit_result.errors['loga_t']:.3f}\n"
            fit_rslt_str += r"$A_\text{loose} = $"
            fit_rslt_str += f"{fit_result.values['loga_l']:.3f} ± "
            fit_rslt_str += f"{fit_result.errors['loga_l']:.3f}\n"
            fit_rslt_str += r"$B = $"
            fit_rslt_str += f"{fit_result.values['logb']:.3f} ± "
            fit_rslt_str += f"{fit_result.errors['logb']:.3f}\n"
        elif self.fit_function == "binomial":
            fit_rslt_str += r"$A_\text{tight} = $"
            fit_rslt_str += f"{fit_result.values['loga_t']:.3f} ± "
            fit_rslt_str += f"{fit_result.errors['loga_t']:.3f}\n"
            fit_rslt_str += r"$A_\text{loose} = $"
            fit_rslt_str += f"{fit_result.values['loga_l']:.3f} ± "
            fit_rslt_str += f"{fit_result.errors['loga_l']:.3f}\n"
            fit_rslt_str += r"$B = $"
            fit_rslt_str += f"{fit_result.values['logb']:.3f} ± "
            fit_rslt_str += f"{fit_result.errors['logb']:.3f}\n"
            fit_rslt_str += r"$N = $"
            fit_rslt_str += f"{fit_result.values['N']:.3f} ± "
            fit_rslt_str += f"{fit_result.errors['N']:.3f}\n"
        ax1_left.text(
            0.07,
            0.45,
            fit_rslt_str,
            transform=ax1_left.transAxes,
            verticalalignment="top",
            horizontalalignment="left",
        )

        # Set y-axis limits for top two plots
        ax1_left.set_ylim(
            10 ** math.floor(math.log10(0.5 * min_y)),
            10 ** math.ceil(math.log10(2 * max_y)),
        )
        ax1_right.set_ylim(
            10 ** math.floor(math.log10(0.5 * min_y)),
            10 ** math.ceil(math.log10(2 * max_y)),
        )
        plt.show()

    def plot_overlay(
        self,
        regions: Optional[list] = ["SR_high_temp", "SR_low_temp"],
        syst: str = "",
        add_label: bool = False,
        add_text: str = "",
    ) -> None:
        """
        Plot an overlay of the histograms for the loose reion, the tight region,
        and the extrapolation of the tight region for the specified regions.

        Parameters
        ----------
        regions : list
            List of regions to plot. E.g. ["SR_high_temp", "SR_low_temp"].
        syst : str
            The systematic variation to use. E.g. "TrkEff". Default is no variation.
        add_label : bool
            Flag to indicate if the CMS label & lumi should be added to the plot.
        add_text : str
            Text to add to the plot. Useful for adding the MC sample name or data label.
        """
        if regions is None:
            print("Warning: No regions specified")
            return

        if not syst.startswith("_") and syst != "":
            syst = f"_{syst}"

        # Find max and min y values for all regions
        max_y = 1
        min_y = 1e8
        for region in regions:
            for subregion in ["_loose", "_tight"]:
                for suffix in ["", "_extrapolation"]:
                    values = self.plots[f"{region}{subregion}{suffix}{syst}"].values()
                    if len(values) == 0:
                        continue
                    values = values[values > 0]
                    max_y = max(max_y, values.max())
                    min_y = min(min_y, values.min())

        # Create figure
        fig = plt.figure(figsize=(18 * len(regions), 10))

        # Create the GridSpec layouts
        gs = gridspec.GridSpec(5, len(regions), left=0.08, right=0.47, bottom=0.15)

        # Create subplots for each region
        for i, region in enumerate(regions):
            ax1 = plt.subplot(gs[0:3, i])  # Top 3 rows
            ax2 = plt.subplot(gs[3:5, i], sharex=ax1)  # 4th row

            h_l_pre = self.plots[f"{region}_loose{syst}"]
            h_l_post = self.plots[f"{region}_loose_extrapolation{syst}"]
            h_t_pre = self.plots[f"{region}_tight{syst}"]
            h_t_post = self.plots[f"{region}_tight_extrapolation{syst}"]

            h_l_pre.plot(
                yerr=np.sqrt(h_l_pre.variances()),
                label="loose",
                color="C0",
                ax=ax1,
            )
            h_l_post.plot(
                yerr=np.sqrt(h_l_post.variances()),
                label="loose extr.",
                color="C0",
                ls="--",
                ax=ax1,
            )
            h_t_pre.plot(
                yerr=np.sqrt(h_t_pre.variances()),
                label="tight",
                color="C1",
                ax=ax1,
            )
            h_t_post.plot(
                yerr=np.sqrt(h_t_post.variances()),
                label="tight extr.",
                color="C1",
                ls="--",
                ax=ax1,
            )
            ax1.set_title(region.replace("_", " ").replace("temp", "T"))
            ax1.set_yscale("log")
            ax1.set_ylim(
                10 ** math.floor(math.log10(0.5 * min_y)),
                10 ** math.ceil(math.log10(2 * max_y)),
            )
            ax1.set_xlabel("")
            ax1.set_ylabel("Events")
            ax1.xaxis.set_minor_locator(ticker.NullLocator())
            ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
            if add_label:
                hep.cms.label(llabel="Preliminary", data=True, lumi=59.8, ax=ax1)
            ax1.legend()

            if add_text != "":
                ax1.text(
                    0.5,
                    0.9,
                    add_text,
                    transform=ax1.transAxes,
                    verticalalignment="top",
                    horizontalalignment="center",
                )

            x_vals = h_l_pre.axes[0].edges
            ratio_loose = np.divide(
                h_l_post.values(),
                h_l_pre.values(),
                where=h_l_pre.values() != 0,
                out=np.zeros_like(h_l_post.values()),
            )
            ratio_err_loose = np.sqrt(
                np.divide(
                    h_l_post.variances(),
                    h_l_pre.values() ** 2,
                    where=h_l_pre.values() != 0,
                    out=np.zeros_like(h_l_post.variances()),
                )
                + np.divide(
                    h_l_post.values() ** 2 * h_l_pre.variances(),
                    h_l_pre.values() ** 4,
                    where=h_l_pre.values() != 0,
                    out=np.zeros_like(h_l_pre.variances()),
                )
            )
            hep.histplot(
                ratio_loose,
                x_vals,
                yerr=ratio_err_loose,
                color="C0",
                linestyle="--",
                lw=2,
                label="loose extr. / loose",
            )
            ratio_tight = np.divide(
                h_t_post.values(),
                h_t_pre.values(),
                where=h_t_pre.values() != 0,
                out=np.zeros_like(h_t_post.values()),
            )
            ratio_err_tight = np.sqrt(
                np.divide(
                    h_t_post.variances(),
                    h_t_pre.values() ** 2,
                    where=h_t_pre.values() != 0,
                    out=np.zeros_like(h_t_post.variances()),
                )
                + np.divide(
                    h_t_post.values() ** 2 * h_t_pre.variances(),
                    h_t_pre.values() ** 4,
                    where=h_t_pre.values() != 0,
                    out=np.zeros_like(h_t_pre.variances()),
                )
            )
            hep.histplot(
                ratio_tight,
                x_vals,
                yerr=ratio_err_tight,
                color="C1",
                linestyle="--",
                lw=2,
                label="tight extr. / tight",
            )
            # ratio_pre = np.divide(
            #     h_t_pre.values() / h_t_pre.sum().value,
            #     h_l_pre.values() / h_l_pre.sum().value,
            #     where=h_l_pre.values() != 0,
            #     out=np.zeros_like(h_t_pre.values()),
            # )
            # ratio_err_pre = np.sqrt(
            #     np.divide(
            #         h_l_pre.variances(),
            #         h_t_pre.values() ** 2,
            #         where=h_t_pre.values() != 0,
            #         out=np.zeros_like(h_l_pre.variances()),
            #     )
            #     + np.divide(
            #         h_l_pre.values() ** 2 * h_t_pre.variances(),
            #         h_t_pre.values() ** 4,
            #         where=h_t_pre.values() != 0,
            #         out=np.zeros_like(h_t_pre.variances()),
            #     )
            # )
            # hep.histplot(
            #     ratio_pre,
            #     x_vals,
            #     yerr=ratio_err_pre,
            #     color="black",
            #     linestyle="--",
            #     lw=2,
            #     label="tight / loose",
            # )
            ax2.legend(ncols=1, loc="upper right")
            ax2.axhline(1, ls="--", color="gray")
            ax2.set_xlabel(r"$n_{muon}$")
            ax2.set_ylabel("ratio")
            ax2.set_ylim(0, 2)
            for label in ax1.xaxis.get_ticklabels():
                label.set_visible(False)

        plt.show()


def convert_strcat_hist_to_root(hist_in, name, axis_name):
    """
    This resolves an issue with uproot's conversion that has a bug.
    It puts the first value to bin 0 (underflow bin) instead of bin 1.
    """
    h_out = ROOT.TH1D(name, axis_name, len(hist_in.values()), 0, len(hist_in.values()))
    for i in range(len(hist_in.values())):
        h_out.SetBinContent(i + 1, hist_in[i].value)
        h_out.SetBinError(i + 1, np.sqrt(hist_in[i].variance))
    return h_out.Clone()


def convert_to_root(
    sample, plots_in, extrapolation=False, do_syst=False, verbose=False
):
    """
    Convert hist.Hist histograms to pyROOT histograms.

    Parameters
    ----------
    sample : str
        Name of the sample.
    plots_in : dict
        Dictionary of hist.Hist histograms.
    extrapolation : bool
        Flag to indicate if extrapolated histograms should be used.
    do_syst : bool
        Flag to indicate if systematic variations should be used.
    verbose : bool
        Flag to indicate if the systematic variations should be printed.

    Returns
    -------
    dict
        Dictionary of pyROOT histograms.
    """
    suffix = ""
    if extrapolation:
        suffix = "_extrapolation"
    plots_out = {}

    systematic_vars = {""}
    for region in plots_in:
        if do_syst and f"SR_high_temp_tight{suffix}_" in region:
            systematic_vars.add(region.replace(f"SR_high_temp_tight{suffix}", ""))
        if do_syst and f"SR_low_temp_tight{suffix}_" in region:
            systematic_vars.add(region.replace(f"SR_low_temp_tight{suffix}", ""))

    systematic_vars = list(systematic_vars)
    if verbose:
        print("Systematic variations:")
        print(systematic_vars)

    for syst in systematic_vars:
        CR_cb_plot = (
            plots_in["CR_cb"]
            if f"CR_cb{syst}" not in plots_in
            else plots_in[f"CR_cb{syst}"]
        )
        plots_out[f"CR_QCD{syst}"] = uproot.to_writable(CR_cb_plot).to_pyroot()  # type: ignore[attr-defined]
        plots_out[f"CR_QCD{syst}"].SetName(f"nMuon_CR_QCD{syst}_{sample}")

        CR_prompt_plot = (
            plots_in["CR_prompt"]
            if f"CR_prompt{syst}" not in plots_in
            else plots_in[f"CR_prompt{syst}"]
        )
        plots_out[f"CR_DY{syst}"] = uproot.to_writable(CR_prompt_plot).to_pyroot()  # type: ignore[attr-defined]
        plots_out[f"CR_DY{syst}"].SetName(f"nMuon_CR_DY{syst}_{sample}")

        # CR_prompt_plot = (
        #     plots_in["CR_prompt"]
        #     if f"CR_prompt{syst}" not in plots_in
        #     else plots_in[f"CR_prompt{syst}"]
        # )
        # h_CR_prompt = ROOT.TH1D(f"nMuon_CR_DY{syst}_{sample}", "nMuon", 1, 2, 3)
        # h_CR_prompt.SetBinContent(1, CR_prompt_plot[2j].value)
        # h_CR_prompt.SetBinError(1, np.sqrt(CR_prompt_plot[2j].variance))
        # plots_out[f"CR_DY{syst}"] = h_CR_prompt.Clone()

        # This for the combined CR (deprecated)
        if "CR" in plots_in:
            CR_plot = (
                plots_in["CR"] if f"CR{syst}" not in plots_in else plots_in[f"CR{syst}"]
            )
            plots_out[f"CR{syst}"] = convert_strcat_hist_to_root(
                CR_plot, f"nMuon_CR{syst}_{sample}", "nMuon"
            )

        if f"SUEP_high_temp{suffix}" in plots_in:
            CR_plot = (
                plots_in[f"SUEP_high_temp{suffix}"]
                if f"SUEP_high_temp{suffix}{syst}" not in plots_in
                else plots_in[f"SUEP_high_temp{suffix}{syst}"]
            )
            plots_out[f"SUEP_high_temp{syst}"] = convert_strcat_hist_to_root(
                CR_plot, f"nMuon_SUEP_high_temp{syst}_{sample}", "nMuon"
            )

        if f"SUEP_low_temp{suffix}" in plots_in:
            CR_plot = (
                plots_in[f"SUEP_low_temp{suffix}"]
                if f"SUEP_low_temp{suffix}{syst}" not in plots_in
                else plots_in[f"SUEP_low_temp{suffix}{syst}"]
            )
            plots_out[f"SUEP_low_temp{syst}"] = convert_strcat_hist_to_root(
                CR_plot, f"nMuon_SUEP_low_temp{syst}_{sample}", "nMuon"
            )

        if f"SR_low_temp_tight{suffix}" in plots_in:
            SR_low_temp_plot = (
                plots_in[f"SR_low_temp_tight{suffix}"]
                if f"SR_low_temp_tight{suffix}{syst}" not in plots_in
                else plots_in[f"SR_low_temp_tight{suffix}{syst}"]
            )
            h_SR_low_temp = ROOT.TH1D(
                f"nMuon_SR_low_temp{syst}_{sample}", "nMuon", 1, 7, 8
            )
            h_SR_low_temp.SetBinContent(1, SR_low_temp_plot[7j].value)
            h_SR_low_temp.SetBinError(1, np.sqrt(SR_low_temp_plot[7j].variance))
            plots_out[f"SR_low_temp{syst}"] = h_SR_low_temp.Clone()

        if f"SR_high_temp_tight{suffix}" in plots_in:
            SR_high_temp_plot = (
                plots_in[f"SR_high_temp_tight{suffix}"]
                if f"SR_high_temp_tight{suffix}{syst}" not in plots_in
                else plots_in[f"SR_high_temp_tight{suffix}{syst}"]
            )
            h_SR_high_temp = ROOT.TH1D(
                f"nMuon_SR_high_temp{syst}_{sample}", "nMuon", 1, 7, 8
            )
            h_SR_high_temp.SetBinContent(1, SR_high_temp_plot[7j].value)
            h_SR_high_temp.SetBinError(1, np.sqrt(SR_high_temp_plot[7j].variance))
            plots_out[f"SR_high_temp{syst}"] = h_SR_high_temp.Clone()

    return plots_out


def export_histograms_to_root(plots, output_path, output_name="output.root"):
    """
    Export hist.Hist histograms to a ROOT file, organized in TDirectories by region.
    Negative bin entries are set to zero.

    Parameters:
    -----------
    plots : dict
        Nested dictionary containing hist.Hist objects
    output_path : str
        Name of the output directory for the ROOT files
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    systematics = [
        syst.replace("SR_high_temp", "")
        for syst in plots["QCD_13TeV_2018"]
        if "SR_high_temp_" in syst
    ]

    with uproot.recreate(os.path.join(output_path, output_name)) as f:
        for sample_name, regions in track(plots.items(), description="Exporting..."):
            for region_name, histogram in regions.items():
                syst_name = ""
                for syst in systematics:
                    if syst in region_name:
                        region_name = region_name.replace(syst, "")
                        syst_name = syst
                        break
                sample_name = sample_name.replace("_13TeV_2018", "")
                f[f"{region_name}/{sample_name}{syst_name}_13TeV_2018"] = (
                    uproot.from_pyroot(histogram)
                )
                # if add_null_obs and syst_name == "" and sample_name == "QCD":
                #     null_hist = histogram.Clone()
                #     null_hist.Reset()
                #     null_hist.SetName(
                #         null_hist.GetName().replace("QCD_Pt_MuEnrichedPt5", "data_obs")
                #     )
                #     f[f"{region_name}/data_obs_13TeV_2018"] = uproot.from_pyroot(
                #         null_hist
                #     )
