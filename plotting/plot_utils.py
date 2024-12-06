import glob
import os
import pickle
import re
from typing import Optional

import dataset_groups
import hist
import matplotlib.pyplot as plt  # type: ignore[import]
import mplhep as hep
import numpy as np
from iminuit import Minuit
from iminuit.cost import LeastSquares
from jacobi import propagate  # type: ignore[import]
from rich.pretty import pprint  # type: ignore[import]

# https://twiki.cern.ch/twiki/bin/viewauth/CMS/RA2b13TeVProduction#Dataset_luminosities_2016_pb_1
lumis = {
    "2016_apv": 19497.914,
    "2016": 16810.813,
    "2017": 41471.589,
    # NOTE: 2018 lumi is only for the main trigger path
    "2018": 54540.000,
}


def lumi_Label(year: str) -> float:
    if year == "2016":
        return round((lumis[year] + lumis[year + "_apv"]) / 1000, 1)
    return round(lumis[year] / 1000, 1)


def find_lumi(
    infile_name: str,
    auto_lumi: Optional[bool] = True,
    year: Optional[str | None] = None,
) -> float:
    if auto_lumi:
        if "20UL16MiniAODv2" in infile_name:
            lumi = lumis["2016"]
        if "20UL17MiniAODv2" in infile_name:
            lumi = lumis["2017"]
        if "20UL16MiniAODAPVv2" in infile_name:
            lumi = lumis["2016_apv"]
        if "20UL18" in infile_name:
            lumi = lumis["2018"]
        if "SUEP-m" in infile_name or "SUEP_m" in infile_name:
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
    year: Optional[int | str | None] = None,
    auto_lumi: Optional[bool | None] = None,
    custom_lumi: Optional[float | None] = None,
    is_data: Optional[bool | None] = False,
) -> dict:
    if isinstance(year, int):
        year = str(year)

    plots_ = {}
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
            plots_[sample] = pickle.load(f)
        for plot in plots_[sample]:
            plots_[sample][plot] = plots_[sample][plot] * lumi

    # Create combined histograms
    for combined_dataset in dataset_groups.dataset_groups_new:
        plots_[combined_dataset] = {}
        for pattern in dataset_groups.dataset_groups_new[combined_dataset]:
            for sample in plots_:
                if re.search(pattern, sample):
                    for plot in plots_[sample]:
                        if plot not in plots_[combined_dataset]:
                            plots_[combined_dataset][plot] = plots_[sample][plot].copy()
                        else:
                            plots_[combined_dataset][plot] += plots_[sample][plot]

    return plots_


def loader(
    tag="test",
    custom_lumi=None,
    load_data=False,
    verbosity=0,
):
    # input .pkl files
    plot_dir = f"../../processor_output_files/{tag}_output_histograms/"
    filenames = glob.glob(plot_dir + "*histograms.pkl")

    # separate the files into signal, background, and data
    files_SUEP = [f for f in filenames if ("SUEP" in f) or ("ggHBSMpythia" in f)]
    files_bkg = [
        f
        for f in filenames
        if ("pythia8" in f) and ("SUEP" not in f) and ("ggHBSMpythia" not in f)
    ]
    files_data = [f for f in filenames if ("DoubleMuon" in f)]
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
    def __init__(self, plots: dict, verbose: Optional[bool] = False):
        self.plots = plots
        self.verbose = verbose

    def muon_func_log(self, x, loga, logb):
        """
        This function is the log of a power law function: a * b^-n
        """
        return loga - x * logb

    def muon_func_tight_log(self, x, loga_t, logb):
        """
        Wrapper for muon_func_log. To be used in the fit for the tight region
        """
        return self.muon_func_log(x, loga_t, logb)

    def muon_func_loose_log(self, x, loga_l, logb):
        """
        Wrapper for muon_func_log. To be used in the fit for the loose region
        """
        return self.muon_func_log(x, loga_l, logb)

    def extrapolate(self):
        for region in self.find_extrapolatable_regions():
            m = self.fit(region)
            h0 = self.evaluate_fit(region, m)
            self.plots[f"{region}_tight_extrapolation"] = h0

    def find_extrapolatable_regions(self):
        return [
            region.replace("_tight", "")
            for region in self.plots
            if "tight" in region and "extrapolation" not in region
        ]

    def get_values(self, h):
        if isinstance(h, hist.Hist):
            return h.values()
        elif isinstance(h, hist.accumulators.WeightedSum):
            return np.array([h.value])
        else:
            raise TypeError(f"{h} must be a hist.Hist or hist.accumulators.WeightedSum")

    def get_variances(self, h):
        if isinstance(h, hist.Hist):
            return h.variances()
        elif isinstance(h, hist.accumulators.WeightedSum):
            return np.array([h.variance])
        else:
            raise TypeError(f"{h} must be a hist.Hist or hist.accumulators.WeightedSum")

    def fit(self, region):
        """
        Perform simultaneous Least Squares fit to the loose and tight regions.
        """
        h_l = self.plots[f"{region}_loose"][1:4]
        h_t = self.plots[f"{region}_tight"][0]

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

        least_squares = LeastSquares(
            data_x_l, data_y_l, data_yerr_l, self.muon_func_loose_log  # type: ignore[arg-type]
        ) + LeastSquares(
            data_x_t, data_y_t, data_yerr_t, self.muon_func_tight_log  # type: ignore[arg-type]
        )
        m = Minuit(least_squares, loga_t=1, loga_l=1, logb=1)
        m.migrad()
        m.hesse()

        if self.verbose:
            print(m)

        return m

    def evaluate_fit(self, region, m):
        """
        This function accepts a Minuit fit result.
        It will output a histogram that represents an evaluation of the fit
        """
        h0 = self.plots[f"{region}_tight"].copy().reset()
        x = h0.axes[0].edges[:-1]
        params_tight = np.array(m.values["loga_t", "logb"])
        cov_tight = np.array(
            [
                [m.covariance["loga_t", "loga_t"], m.covariance["loga_t", "logb"]],
                [m.covariance["logb", "loga_t"], m.covariance["logb", "logb"]],
            ]
        )
        logy, logycov = propagate(
            lambda p: self.muon_func_log(x - 3, *p), params_tight, cov_tight
        )
        logyerr_prop = np.diag(logycov) ** 0.5
        y = 10**logy
        yerr_prop = np.log(10) * y * logyerr_prop
        for i in range(len(h0.values())):
            h0[i] = (y[i], yerr_prop[i] ** 2)
        return h0

    def plot_fit(self, plots, values, covariance, histname="SR_high_temp_tight"):
        h0 = plots[histname]

        fig = plt.figure(figsize=(12, 12))
        plt.subplots_adjust(bottom=0.15, left=0.17)
        ax1 = plt.subplot2grid((5, 1), (0, 0), rowspan=3)
        ax2 = plt.subplot2grid((5, 1), (3, 0), sharex=ax1)
        ax3 = plt.subplot2grid((5, 1), (4, 0), sharex=ax1)

        x = np.arange(3, 8)
        logy, logycov = propagate(
            lambda p: self.muon_func_log(x - 4, *p), values, covariance
        )
        logyerr_prop = np.diag(logycov) ** 0.5

        y = 10**logy
        yerr_prop = np.log(10) * y * logyerr_prop

        x_hatch = np.vstack((np.arange(3, 8), np.arange(4, 9))).reshape(
            (-1,), order="F"
        )
        y_hatch = np.vstack((y, y)).reshape((-1,), order="F")
        y_hatch_unc = np.vstack((yerr_prop, yerr_prop)).reshape((-1,), order="F")
        hep.histplot(h0, yerr=np.sqrt(h0.variances()), label="MC", ax=ax1)
        hep.histplot(y, bins=np.arange(3, 9), label="fit", ax=ax1)
        ax1.fill_between(
            x=x_hatch,
            y1=y_hatch - y_hatch_unc,  # type: ignore[arg-type]
            y2=y_hatch + y_hatch_unc,  # type: ignore[arg-type]
            label="Stat. Unc.",
            step="pre",
            facecolor="C1",
            alpha=0.3,
            linewidth=0,
        )
        ax1.legend()
        ax1.set_yscale("log")
        ax1.set_xlabel("")
        ax1.set_ylabel("Events")
        ax1.set_xlim(2.8, 8.2)
        # ax1.axes.xaxis.set_ticklabels([])
        # ax1.set_xticklabels([])

        ratio = h0.values() / y
        ratio_err = np.sqrt(
            (y**-2) * (h0.variances()) + (h0.values() ** 2 * y**-4) * (yerr_prop**2)
        )
        ax2.errorbar(
            h0.axes.centers[0],
            ratio,
            yerr=ratio_err,
            color="black",
            fmt="o",
            linestyle="none",
        )
        ax2.axhline(1, ls="--", color="gray")
        ax2.set_xlabel("")
        ax2.set_ylabel("MC / fit")
        ax2.set_xlim(2.8, 8.2)
        ax2.set_ylim(0, 2)
        # ax2.set_xticklabels([])

        pulls = (h0.values() - y) / np.sqrt(h0.variances())
        pulls_up = np.where(pulls >= 0, pulls, 0)
        pulls_down = np.where(pulls < 0, pulls, 0)

        x_hatch = np.vstack((np.arange(3, 8), np.arange(4, 9))).reshape(
            (-1,), order="F"
        )
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

        temp = ax1.get_xticklabels()
        # ax1.set_xticklabels([])
        ax3.set_xticklabels(temp)

        plt.show()
