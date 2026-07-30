# %%

from phase1 import run_phase1_analysis
from phase2 import run_phase2_analysis
from phase3 import run_phase3_analysis

subject = "CB072"

for method in ("mag", "grad", "eeg"):
    run_phase1_analysis(subject=subject, method=method)

for method in ("mag", "grad", "eeg"):
    run_phase2_analysis(subject=subject, method=method)



for dur in ("500", "1000", "1500")
    for method in ("mag", "grad", "eeg"):
        
        epochs = mne.read_epochs(
            rf"{file_path}\Phase3_offset\{subject}_04_epochs_offset_{method}_offset{dur}_epo.fif",
            preload=False,
        )    
        
        run_phase3_analysis(epochs=epochs, subject=subject, method=method)

# %%
