import json
import os
import subprocess

dir_path = (
    "/store/user/lpcsuep/SUEPNano_data/DoubleMuon+Run2018A-UL2018_MiniAODv2-v1+MINIAOD"
)
xrootd_redirector = "root://cmseos.fnal.gov/"

file_dict = {}
file_dict["DoubleMuon+Run2018A-UL2018_MiniAODv2-v1+MINIAOD"] = []
result = subprocess.run(
    ["eos", xrootd_redirector, "ls", dir_path], stdout=subprocess.PIPE
)
files = result.stdout.decode("utf-8").split("\n")
for _file in files:
    if _file:  # ignore empty lines
        file_dict["DoubleMuon+Run2018A-UL2018_MiniAODv2-v1+MINIAOD"].append(
            os.path.join(xrootd_redirector + dir_path, _file)
        )

# Write the dictionary to a JSON file
with open("filelist.json", "w") as f:
    json.dump(file_dict, f, indent=4)
