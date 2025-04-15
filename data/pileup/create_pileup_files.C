#include <TFile.h>
#include <TH1.h>

void create_pileup_files(
    const std::string& central_file,
    const std::string& up_file,
    const std::string& down_file,
    const std::string& output_file
) {
    // Open the input files
    TFile* central = TFile::Open(central_file.c_str());
    TFile* up = TFile::Open(up_file.c_str());
    TFile* down = TFile::Open(down_file.c_str());

    // Check if the files were opened successfully
    if (!central || !up || !down) {
        std::cerr << "Error opening input files." << std::endl;
        return;
    }
    // Create the output file
    TFile* output = TFile::Open(output_file.c_str(), "RECREATE");
    if (!output) {
        std::cerr << "Error creating output file." << std::endl;
        return;
    }
    // Create histograms for the pileup distributions
    TH1F* h_central = (TH1F*)central->Get("pileup");
    TH1F* h_up = (TH1F*)up->Get("pileup");
    TH1F* h_down = (TH1F*)down->Get("pileup");

    h_up->SetName("pileup_up");
    h_down->SetName("pileup_down");

    // Write the histograms to the output file
    output->cd();
    h_central->Write();
    h_up->Write();
    h_down->Write();
    // Close the files
    central->Close();
    up->Close();
    down->Close();
    output->Close();

    return;
}
