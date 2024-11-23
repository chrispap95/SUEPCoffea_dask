import json

# Load the xsec data
with open("xsections_2018.json") as f:
    xsecs = json.load(f)

# Print the xsec data
xsec_new = {}
for dataset in xsecs:
    xsec_new[dataset] = {"xsec": xsecs[dataset], "br": 1.0}

# Save the xsec data
with open("xsections_2018.json", "w") as f:
    json.dump(xsec_new, f, indent=4)
