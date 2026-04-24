
#The almighty file 
from pathlib import Path
from paths import create_output_folders
from step00_badch_maxwell import run_badch_maxwell
from step01_prep_pipeline import run_prep_pipeline
from step02_artifact_annotations import run_artifact_annotations
from step03_ica import run_train_ica, run_apply_ica



inroot_dir = Path(r"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE")

subject = "CA140"

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






def run_full_pipeline_part1(subject):

    # -------------------------
    # PATHS
    # -------------------------
    inroot = Path(r"C:\Users\tomas\Desktop\COG_MEEG_EXP1_RELEASE")
    out_paths = create_output_folders(subject=subject, inroot=inroot)

    sub_indir = inroot / subject
    sub_dur_indir = sub_indir / f"{subject}_EXP1_MEEG"

    names = [f"dur{i}" for i in range(1, 6)]
    
    # -------------------------
    # FILES RAW
    # -------------------------
    raw_files = sorted([x for x in sub_dur_indir.glob("*DurR*.fif")])
    """
    # -------------------------
    # STEP 00: Maxwell
    # -------------------------
    raws_sss = run_badch_maxwell(
        file_paths=raw_files,
        cal_file = sub_indir / "metadata/calibration_crosstalk_coreg" / f"{subject}_ses-1_acq-calibration_meg.dat",
        ct_file  = sub_indir / "metadata/calibration_crosstalk_coreg" / f"{subject}_ses-1_acq-crosstalk_meg.fif",
        out_paths=out_paths,
        subject=subject,
        names=names,
        ,
    )
    
    sss_files = [
        out_paths["00_badch_maxwell"] / f"{subject}_badch_maxwell_{n}.fif"
        for n in names
    ]

    # -------------------------
    # STEP 01: PREP
    # -------------------------
    raws_clean = run_prep_pipeline(
        file_paths=sss_files,
        out_paths=out_paths,
        subject=subject,
        names=names,
    )

    prep_files = [
        out_paths["01_prep_pipeline"] / f"{subject}_01_prep_pipeline_{n}.fif"
        for n in names
    ]

    # -------------------------
    # STEP 02: ANNOTATIONS
    # -------------------------
    raws_annotated = run_artifact_annotations(
        file_paths=prep_files,
        out_paths=out_paths,
        subject=subject,
        names=names,
    )
"""    
    annot_files = [
        out_paths["02_artifact_annotations"] / f"{subject}_02_artifact_annotations_{n}.fif"
        for n in names
    ]

    # -------------------------
    # STEP 03: ICA (train)
    # -------------------------
    ica_meg, ica_eeg = run_train_ica(
        file_paths=annot_files,
        out_paths=out_paths,
        subject=subject,
        names=names,
    )


if __name__ == "__main__":
    run_full_pipeline_part1("CA140")





"""

    print("\n👉 Agora corre manualmente o inspect_ica antes de aplicar.")

    input("Press Enter depois de escolheres componentes...")

    # -------------------------
    # STEP 04: ICA APPLY
    # -------------------------
    raw_final = run_apply_ica(
        file_paths=annot_files,
        out_paths=out_paths,
        subject=subject,
        names=names,
    )

    print("Pipeline completo.")

    return raw_final

"""




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

#*#*#*#*#*#*#*#*#*#*#*#*#*#
#   1) Analise daquilo    #
#*#*#*#*#*#*#*#*#*#*#*#*#*#

#-->
# %%
