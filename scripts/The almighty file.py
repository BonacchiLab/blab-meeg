#The almighty file 


from pathlib import Path
from paths import create_output_folder 

"""
from 00_badch_maxwell import run_badch_maxwell
from 01_prep_pipeline import run_prep_pipeline
from 02_artifact_annotations import run_artifact_annotations
from 03_ica import run_ica
from 04_epochs import run_preprocess_epochs
"""
import run_badch_maxwell
import run_prep_pipeline
import run_artifact_annotations
import run_ica
import run_preprocess_epochs


inroot_dir = Path(r"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE")
subject = "CA124"

sub_indir = Path(fr"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE\{subject}")
sub_dur_indir = sub_indir / f"{subject}_EXP1_MEEG"

out_paths = create_output_folders(subject=subject, inroot=inroot_dir)

outroot_dir = r"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE_OUTPUT"

    # --- caminhos dos ficheiros ---
file_paths = [
    fr"{sub_dur_indir}\{subject}_MEEG_1_DurR1.fif",
    fr"{sub_dur_indir}\{subject}_MEEG_1_DurR2.fif",
    fr"{sub_dur_indir}\{subject}_MEEG_1_DurR3.fif",
    fr"{sub_dur_indir}\{subject}_MEEG_1_DurR4.fif",
    fr"{sub_dur_indir}\{subject}_MEEG_1_DurR5.fif"
]
names = ["dur1", "dur2", "dur3", "dur4", "dur5"]
dur_files = [sub_dur_indir / f"{subject}_MEEG_1_DurR{i}.fif" for i in range(1,6)]
dur_files = [x for x in sub_dur_indir.glob("*") if x.suffix == ".fif" and "DurR" in x.name]

# ficheiros de calibração e cross-talk
cal_file = fr"{sub_indir}\metadata\calibration_crosstalk_coreg\{subject}_ses-1_acq-calibration_meg.dat"
ct_file = fr"{sub_indir}\metadata\calibration_crosstalk_coreg\{subject}_ses-1_acq-crosstalk_meg.fif"

raws_sss = run_badch_maxwell(
    file_paths=file_paths,
    cal_file=cal_file,
    ct_file=ct_file,
    out_paths=out_paths,
    subject=subject,
    names=names,
)

raws_prepped = run_prep_pipeline(
    file_paths=file_paths,
    out_paths=out_paths,
    subject=subject,
    names=names,
)

raws_annotated = run_artifact_annotations(
    file_paths=file_paths,
    out_paths=out_paths,
    subject=subject,
    names=names,
)

raw_concatenated = run_ica(
    file_paths=file_paths,
    out_paths=out_paths,
    subject=subject,
    names=names,
)

epochs_clean = run_preprocess_epochs(
    file_paths=file_paths[0],
    out_paths=out_paths,
    subject=subject,
)







#========================#
#===1) Analise daquilo===#
#========================#

#-~-~-~-~-~-~-~-~-~-~-~-#
#-~1)-Analise daquilo-~-#
#-~-~-~-~-~-~-~-~-~-~-~-#

#<><><><><><><><><><><><>#
#<>1)<>Analise<>daquilo<>#
#<><><><><><><><><><><><>#

#´`´`´`´`´`´`´`´`´`´`´`´`#
#´`1)´`Analise´`daquilo´`#
#´`´`´`´`´`´`´`´`´`´`´`´`#

#´`´`´`´`´`´`´`´`´`´`´`´`#
#´`1)´`Analise´`daquilo´`#
#________________________#

#_______________________#
#                       #
#   1) Analise daquilo  #
#_______________________#
#                       #

#()()()()()()()()()()()()#
#()1)()Analise()daquilo()#
#()()()()()()()()()()()()#

#||||||||||||||||||||||||#
#||1)||Analise||daquilo||#
#||||||||||||||||||||||||#

##########################
###1)##Analise##daquilo###
##########################

#========================#
#==1)==Analise==daquilo==#
#========================#

#''''''''''''''''''''''''#
#''1)''Analise''daquilo''#
#''''''''''''''''''''''''#

#''''''''''''''''''''''''#
#--1)--Analise--daquilo--#
#........................#

#-+-+-+-+-+-+-+-+-+-+-+-+-+-#
#-+-1-+-Analise-+-daquilo-+-#
#-+-+-+-+-+-+-+-+-+-+-+-+-+-#

#{{{{{[[[[[((((()))))]]]]]}}}}}#
#{{[[((1)-Analise-daquilo))]]}}#
#{{{{{[[[[[((((()))))]]]]]}}}}}#

#*#*#*#*#*#*#*#*#*#*#*#*#*#
#*#1#*#Analise#*#daquilo#*#
#*#*#*#*#*#*#*#*#*#*#*#*#*#