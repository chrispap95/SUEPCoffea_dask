import json

from rich import print  # type: ignore[import]


def create_eta_pt_correction(
    name,
    description,
    data,
):
    """
    Create a correctionlib JSON structure for 2D binned data.

    Parameters:
    -----------
    name : str
        Name of the correction
    description : str
        Description of the correction
    data : dict
        Correction values for each bin
        Shape should be (len(eta_edges)-1, len(pt_edges)-1)

    Returns:
    --------
    dict
        Complete correctionlib JSON structure
    """
    content = []
    for eta_bin_data in data["eta_binned_data"]:
        content_per_eta = []
        for pt_bin_data in eta_bin_data["pt_binned_data"]:
            content_per_eta.append(
                {
                    "nodetype": "category",
                    "input": "scale_factors",
                    "content": [
                        {
                            "key": "nominal",
                            "value": pt_bin_data["nominal"],
                        },
                        {
                            "key": "error",
                            "value": pt_bin_data["error"],
                        },
                        {
                            "key": "systdown",
                            "value": pt_bin_data["nominal"] - pt_bin_data["error"],
                        },
                        {
                            "key": "systup",
                            "value": pt_bin_data["nominal"] + pt_bin_data["error"],
                        },
                    ],
                }
            )
        content.append(
            {
                "nodetype": "binning",
                "input": "pt",
                "edges": eta_bin_data["pt_edges"],
                "content": content_per_eta,
                "flow": "error",
            },
        )

    correction_data = {
        "schema_version": 2,
        "description": description,
        "corrections": [
            {
                "name": name,
                "description": name,
                "version": 2,
                "inputs": [
                    {"name": "eta", "type": "real", "description": "Probe eta"},
                    {"name": "pt", "type": "real", "description": "Probe pt"},
                    {
                        "name": "scale_factors",
                        "type": "string",
                        "description": "Choose nominal scale factor or one of the uncertainties",
                    },
                ],
                "output": {
                    "name": "weight",
                    "type": "real",
                    "description": "Output scale factor (nominal) or uncertainty",
                },
                "data": {
                    "nodetype": "transform",
                    "input": "eta",
                    "rule": {
                        "nodetype": "formula",
                        "expression": "abs(x)",
                        "parser": "TFormula",
                        "variables": ["eta"],
                    },
                    "content": {
                        "nodetype": "binning",
                        "input": "eta",
                        "edges": data["eta_edges"],
                        "content": content,
                        "flow": "error",
                    },
                },
            }
        ],
    }

    return correction_data


def parse_old_CMS_JSON(input_json):
    input_json = input_json["NUM_TrackerMuons_DEN_genTracks"]["abseta_pt"]

    data = {}

    # Get eta edges
    eta_edges = set()
    for key in input_json.keys():
        key = key.replace("abseta:[", "").replace("]", "")
        for eta in key.split(","):
            eta_edges.add(float(eta))
    eta_edges = sorted(list(eta_edges))
    data["eta_edges"] = eta_edges

    data["eta_binned_data"] = []
    for i in range(len(eta_edges) - 1):
        eta_bin = f"abseta:[{eta_edges[i]:.2f},{eta_edges[i+1]:.2f}]"
        pt_edges = set()
        for key in input_json[eta_bin].keys():
            key = key.replace("pt:[", "").replace("]", "")
            for pt in key.split(","):
                pt_edges.add(float(pt))
        pt_edges = sorted(list(pt_edges))

        eta_bin_data = {}
        eta_bin_data["pt_edges"] = pt_edges
        eta_bin_data["pt_binned_data"] = []
        for j in range(len(pt_edges) - 1):
            pt_bin = f"pt:[{pt_edges[j]:.2f},{pt_edges[j+1]:.2f}]"
            pt_bin_data = {
                "nominal": input_json[eta_bin][pt_bin]["value"],
                "error": input_json[eta_bin][pt_bin]["error"],
            }
            eta_bin_data["pt_binned_data"].append(pt_bin_data)

        data["eta_binned_data"].append(eta_bin_data)

    return data


if __name__ == "__main__":
    # Input old CMS JSON file
    with open(
        "low_pt_muons/Efficiency_muon_generalTracks_Run2018_UL_trackerMuon.json", "r"
    ) as f:
        old_json = json.load(f)

    # Parse the old JSON file
    data = parse_old_CMS_JSON(old_json)

    # Create the correction
    correction = create_eta_pt_correction(
        name="NUM_TrackerMuons_DEN_genTracks",
        description="Custom corrections file",
        data=data,
    )

    # Save to file
    with open("my_correction.json", "w") as f:
        json.dump(correction, f, indent=4)
