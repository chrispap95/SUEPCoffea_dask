from coffea import lumi_tools


def apply_golden_JSON(events, era):
    if era == "2016" or era == "2016APV":
        LumiJSON = lumi_tools.LumiMask(
            "data/GoldenJSON/Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt"
        )
    elif era == "2017":
        LumiJSON = lumi_tools.LumiMask(
            "data/GoldenJSON/Cert_294927-306462_13TeV_UL2017_Collisions17_GoldenJSON.txt"
        )
    elif era == "2018":
        LumiJSON = lumi_tools.LumiMask(
            "data/GoldenJSON/Cert_314472-325175_13TeV_Legacy2018_Collisions18_JSON.txt"
        )
    elif era == "2022":
        LumiJSON = lumi_tools.LumiMask(
            "data/GoldenJSON/Cert_Collisions2022_355100_362760_Golden.json"
        )
    elif era == "2023":
        LumiJSON = lumi_tools.LumiMask(
            "data/GoldenJSON/Cert_Collisions2023_366442_370790_Golden.json"
        )
    else:
        print("No era is defined. Please specify the year")

    events = events[LumiJSON(events.run, events.luminosityBlock)]

    return events
