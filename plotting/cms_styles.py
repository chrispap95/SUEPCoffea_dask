from cycler import cycler  # type: ignore[import]

cmap_petroff_6 = ["#5790fc", "#f89c20", "#e42536", "#964a8b", "#9c9ca1", "#7a21dd"]
cmap_petroff_10 = [
    "#3f90da",
    "#ffa90e",
    "#bd1f01",
    "#94a4a2",
    "#832db6",
    "#a96b59",
    "#e76300",
    "#b9ac70",
    "#717581",
    "#92dadd",
]

CMS_petroff_6 = {"axes.prop_cycle": cycler("color", cmap_petroff_6)}
CMS_petroff_10 = {"axes.prop_cycle": cycler("color", cmap_petroff_10)}
