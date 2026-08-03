#%%
from pathlib import Path
import shutil
from blab_meeg.utils.paths import create_output_folders
import dicom2nifti

import os
import subprocess


def convert_dicom_to_nifti(subject, paths):
    """
    Converte a MRI em formato DICOM para NIfTI (.nii.gz).

    Parameters
    ----------
    subject : str
        ID do participante.
    paths : dict
        Dicionário devolvido por create_output_folders().

    Returns
    -------
    Path
        Caminho para o ficheiro NIfTI criado.
    """


    # ---------------------------------------------------------
    # Pasta de saída
    # ---------------------------------------------------------

    nifti_folder = paths["freesurfer_input"]
    nifti_folder.mkdir(parents=True, exist_ok=True)

    final_file = nifti_folder / f"{subject}.nii.gz"

    # ---------------------------------------------------------
    # Já existe?
    # ---------------------------------------------------------

    if final_file.exists():
        print(f"{subject}: NIfTI já existe.")
        return final_file


    # ---------------------------------------------------------
    # Pasta onde estão os DICOM
    # ---------------------------------------------------------

    dicom_folder = Path(
        f"/home/blab/COGITATE/DATA/COG_MEEG_EXP1_RELEASE/{subject}/{subject}__MR"
    )

    # ---------------------------------------------------------
    # Guardar os ficheiros existentes
    # ---------------------------------------------------------

    before = set(nifti_folder.glob("*.nii.gz"))

    # ---------------------------------------------------------
    # Conversão
    # ---------------------------------------------------------

    dicom2nifti.convert_directory(
        str(dicom_folder),
        str(nifti_folder),
        compression=True,
        reorient=True,
    )

    # ---------------------------------------------------------
    # Descobrir quais foram criados nesta conversão
    # ---------------------------------------------------------

    after = set(nifti_folder.glob("*.nii.gz"))

    nifti_files = list(after - before)

    if len(nifti_files) == 0:
        raise RuntimeError("Nenhum ficheiro NIfTI foi criado.")

    if len(nifti_files) > 1:
        raise RuntimeError(
            f"Foram encontrados {len(nifti_files)} ficheiros NIfTI."
        )

    # ---------------------------------------------------------
    # Renomear
    # ---------------------------------------------------------

    final_file = nifti_folder / f"{subject}.nii.gz"

    shutil.move(
        str(nifti_files[0]),
        str(final_file),
    )

    return final_file




def run_recon_all(subject, nifti_file, paths):
    """
    Executa o recon-all do FreeSurfer.

    Parameters
    ----------
    subject : str
        ID do participante.
    nifti_file : Path
        MRI T1 em formato NIfTI.
    paths : dict
        Dicionário devolvido por create_output_folders().
    """

    env = os.environ.copy()

    env["SUBJECTS_DIR"] = str(paths["freesurfer"])

    command = [
        "recon-all",
        "-subject",
        subject,
        "-i",
        str(nifti_file),
        "-all",
    ]

    print("\nRunning:")
    print(" ".join(command))
    print()

    subprocess.run(
        command,
        env=env,
        check=True,
    )



if __name__ == "__main__":

    subject = "CA107"

    paths = create_output_folders(subject)

    nifti = convert_dicom_to_nifti(subject, paths)

    run_recon_all(subject, nifti, paths)
# %%


import subprocess
import os

env = os.environ.copy()
env["SUBJECTS_DIR"] = str(paths["freesurfer"])

subprocess.run(
    [
        "mne",
        "watershed_bem",
        "--subject",
        subject,
    ],
    env=env,
    check=True,
)



#%%
import mne


def run_source_and_bem(subject, paths):
    """
    Cria o Source Space e o BEM para um sujeito.

    Parameters
    ----------
    subject : str
        ID do participante.
    paths : dict
        Dicionário devolvido por create_output_folders().
    """

    subjects_dir = paths["freesurfer"]

    bem_folder = subjects_dir / subject / "bem"
    bem_folder.mkdir(parents=True, exist_ok=True)

    # ==========================================================
    # SOURCE SPACE
    # ==========================================================

    src_file = bem_folder / f"{subject}-oct6-src.fif"

    if src_file.exists():

        print(f"{subject}: Source space já existe.")

        src = mne.read_source_spaces(src_file)

    else:

        print(f"{subject}: Creating source space...")

        src = mne.setup_source_space(
            subject=subject,
            spacing="oct6",
            subjects_dir=subjects_dir,
            add_dist=False,
        )

        mne.write_source_spaces(
            src_file,
            src,
            overwrite=True,
        )

    # ==========================================================
    # BEM MODEL
    # ==========================================================

    bem_model_file = bem_folder / f"{subject}-5120-5120-5120-bem.fif"

    bem_solution_file = bem_folder / f"{subject}-5120-5120-5120-bem-sol.fif"

    if bem_solution_file.exists():

        print(f"{subject}: BEM já existe.")

        bem = mne.read_bem_solution(bem_solution_file)

    else:

        print(f"{subject}: Creating BEM model...")

        bem_model = mne.make_bem_model(
            subject=subject,
            ico=4,
            conductivity=(0.3,),
            subjects_dir=subjects_dir,
        )

        mne.write_bem_surfaces(
            bem_model_file,
            bem_model,
            overwrite=True,
        )

        print(f"{subject}: Computing BEM solution...")

        bem = mne.make_bem_solution(bem_model)

        mne.write_bem_solution(
            bem_solution_file,
            bem,
            overwrite=True,
        )

    return src, bem


if __name__ == "__main__":

    subject = "CA107"

    paths = create_output_folders(subject)

    src, bem = run_source_and_bem(subject, paths)

# %%

from pathlib import Path

import mne

from blab_meeg.utils.paths import create_output_folders


def run_coregistration(subject, raw_file, paths):
    """
    Abre a interface gráfica de coregistration.

    Parameters
    ----------
    subject : str
    raw_file : Path
        Raw concatenado.
    paths : dict
    """

    subjects_dir = paths["freesurfer"]

    trans_file = paths["freesurfer_coreg"] / f"{subject}-trans.fif"

    # já existe?
    if trans_file.exists():
        print(f"{subject}: trans.fif já existe.")
        return trans_file

    mne.gui.coregistration(
        subject=subject,
        subjects_dir=subjects_dir,
        inst=raw_file,
    )

    print()
    print("Quando terminares:")
    print("File → Save Transform")
    print(trans_file)

    return trans_file

if __name__ == "__main__":
    subject = "CA107"

    paths = create_output_folders(subject)

    raw_file = (
        paths["03_ica"] /
        f"{subject}_03_ica_concat_raw.fif"
    )

    run_coregistration(
        subject,
        raw_file,
        paths,
    )
#%%  
##############################################################################
# ESTE CODIGO NAO FUNCIONA PRECISA DO TRANS FIF QUE PRECISA DE UM INTERATIVO #
##############################################################################


from pathlib import Path

import mne

from blab_meeg.utils.paths import create_output_folders


subject = "CA107"

paths = create_output_folders(subject)

raw_file = (
    paths["03_ica"] /
    f"{subject}_03_ica_concat_raw.fif"
)


subjects_dir = paths["freesurfer"]

trans_file = paths["freesurfer_coreg"] / f"{subject}-trans.fif"

# já existe?
if trans_file.exists():
    print(f"{subject}: trans.fif já existe.")
  

mne.gui.coregistration(
    subject=subject,
    subjects_dir=subjects_dir,
    inst=raw_file,
)

print()
print("Quando terminares:")
print("File → Save Transform")
print(trans_file)


#%%

#%%
from pathlib import Path

import mne

from blab_meeg.utils.paths import create_output_folders


def make_forward_solution(subject, raw_file, paths):
    """
    Cria a Forward Solution.

    Parameters
    ----------
    subject : str
    raw_file : Path
        Raw concatenado após ICA.
    paths : dict

    Returns
    -------
    Forward
    """

    subjects_dir = paths["freesurfer"]

    src_file = (
        subjects_dir /
        subject /
        "bem" /
        f"{subject}-oct6-src.fif"
    )

    bem_file = (
        subjects_dir /
        subject /
        "bem" /
        f"{subject}-5120-5120-5120-bem-sol.fif"
    )
        

    fwd_file = (
        paths["freesurfer_forward"] /
        f"{subject}-fwd.fif"
    )

    fwd_file.parent.mkdir(parents=True, exist_ok=True)

    if fwd_file.exists():
        print(f"{subject}: Forward já existe.")
        return mne.read_forward_solution(fwd_file)

    raw = mne.io.read_raw_fif(raw_file, preload=False)

    fwd = mne.make_forward_solution(
        info=raw.info,
        trans="fsaverage",          # <-- trocar depois por CA107-trans.fif
        src=src_file,
        bem=bem_file,
        meg=True,
        eeg=False,
        mindist=5.0,
        n_jobs=1,
    )

    mne.write_forward_solution(
        fwd_file,
        fwd,
        overwrite=True,
    )

    return fwd


if __name__ == "__main__":

    subject = "CA107"

    paths = create_output_folders(subject)

    raw_file = (
        paths["03_ica"] /
        f"{subject}_03_ica_concat_raw.fif"
    )

    fwd = make_forward_solution(
        subject,
        raw_file,
        paths,
    )

    print(fwd)
# %%

# Nice ha empy room recording 
import mne

from blab_meeg.utils.paths import create_output_folders

inroot = Path("/home/blab/COGITATE/DATA/COG_MEEG_EXP1_RELEASE")

paths = create_output_folders(subject)

raw = mne.io.read_raw_fif(inroot / subject / f"{subject}_RNoise_MEEG" / f"{subject}_MEEG_1_Rnoise.fif")

print(raw)

# %%
print(raw.info["description"])
print(raw.info["experimenter"])
print(raw.info["meas_date"])
# %%




#%% 
#%%
from pathlib import Path

import mne

from blab_meeg.utils.paths import create_output_folders


def make_inverse_operator(
    subject,
    raw_file,
    epochs_file,
    paths,
):

    inverse_file = (
        paths["freesurfer_subject"]
        / "bem"
        / f"{subject}-meg-inv.fif"
    )

    if inverse_file.exists():
        print(f"{subject}: Inverse operator já existe.")
        return inverse_file

    fwd_file = (
        paths["freesurfer_forward"]
        / f"{subject}-fwd.fif"
    )

    raw = mne.io.read_raw_fif(raw_file, preload=False)

    inroot = Path("/home/blab/COGITATE/DATA/COG_MEEG_EXP1_RELEASE")

    noise_file = inroot / subject / f"{subject}_RNoise_MEEG" / f"{subject}_MEEG_1_Rnoise.fif"


    raw_noise = mne.io.read_raw_fif(
        noise_file,
        preload=True,
    )

    noise_cov = mne.compute_raw_covariance(
        raw_noise
    )

    fwd = mne.read_forward_solution(fwd_file)

    inverse = mne.minimum_norm.make_inverse_operator(
        info=raw.info,
        forward=fwd,
        noise_cov=noise_cov,
        loose=0.2,
        depth=0.8,
    )

    mne.minimum_norm.write_inverse_operator(
        inverse_file,
        inverse,
        overwrite=True,
    )

    print("Inverse operator guardado em:")
    print(inverse_file)

    return inverse_file


if __name__ == "__main__":

    subject = "CA107"

    paths = create_output_folders(subject)

    raw_file = (
        paths["03_ica"]
        / f"{subject}_03_ica_concat_raw.fif"
    )

    epochs_file = (
        paths["phase1_epochs"]
        / f"{subject}_04_epochs_meg_Phase1_epo.fif"
    )

    make_inverse_operator(
        subject,
        raw_file,
        epochs_file,
        paths,
    )
# %%
#%%
from pathlib import Path

import mne

from mne.minimum_norm import (
    read_inverse_operator,
    apply_inverse_epochs,
)

from blab_meeg.utils.paths import create_output_folders


def compute_source_estimates(
    subject,
    epochs_file,
    inverse_file,
    paths,
):

    output_folder = (
        paths["freesurfer_subject"]
        / "stcs"
    )

    output_folder.mkdir(
        exist_ok=True,
        parents=True,
    )

    epochs = mne.read_epochs(
        epochs_file,
        preload=True,
    )

    inverse = read_inverse_operator(
        inverse_file
    )

    stcs = apply_inverse_epochs(
        epochs,
        inverse,
        lambda2=1 / 9,
        method="dSPM",
        pick_ori=None,
    )

    for i, stc in enumerate(stcs):

        stc.save(
            output_folder
            / f"{subject}_{i:04d}"
        )

    print(f"{len(stcs)} source estimates guardadas.")

    return stcs


if __name__ == "__main__":

    subject = "CA107"

    paths = create_output_folders(subject)

    epochs_file = (
        paths["phase1_epochs"]
        / f"{subject}_04_epochs_meg_Phase1_epo.fif"
    )

    inverse_file = (
        paths["freesurfer_subject"]
        / "bem"
        / f"{subject}-meg-inv.fif"
    )

    compute_source_estimates(
        subject,
        epochs_file,
        inverse_file,
        paths,
    )
# %%

# %%
from pathlib import Path

import mne

from blab_meeg.utils.paths import create_output_folders


def explore_single_stc(
    subject,
    epochs_file,
    inverse_file,
    paths,
    epoch_index=0,
    method="dSPM",
):

    subjects_dir = paths["freesurfer"]

    print("Loading epochs...")
    epochs = mne.read_epochs(
        epochs_file,
        preload=True,
    )

    print("Loading inverse operator...")
    inverse = mne.minimum_norm.read_inverse_operator(
        inverse_file
    )

    print(f"\nComputing STC for epoch {epoch_index}")

    stc = mne.minimum_norm.apply_inverse(
        epochs[epoch_index].average(),
        inverse,
        lambda2=1.0 / 9.0,
        method=method,
        pick_ori=None,
    )

    print("\nOpening interactive brain...")

    brain = stc.plot(
        subject=subject,
        subjects_dir=subjects_dir,
        hemi="both",
        surface="inflated",
        cortex="low_contrast",
        views="lat",
        time_viewer=True,
        clim="auto",
    )

    return brain


if __name__ == "__main__":

    subject = "CA107"

    paths = create_output_folders(subject)

    epochs_file = (
        paths["phase1_epochs"]
        / f"{subject}_04_epochs_meg_Phase1_epo.fif"
    )
    
    inverse_file = (
        paths["freesurfer"]
        / subject
        / "bem"
        / f"{subject}-meg-inv.fif"
    )

    brain = explore_single_stc(
        subject,
        epochs_file,
        inverse_file,
        paths,
        epoch_index=0,
        method="dSPM",
    )
# %%
# %%
from pathlib import Path

import mne
import numpy as np

from blab_meeg.utils.paths import create_output_folders


def explore_average_stc(
    subject,
    epochs_file,
    inverse_file,
    paths,
    condition="faces",
    method="sLORETA",
):

    subjects_dir = paths["freesurfer"]

    print("Loading epochs...")
    epochs = mne.read_epochs(
        epochs_file,
        preload=True,
    )

    print("Loading inverse operator...")
    inverse = mne.minimum_norm.read_inverse_operator(
        inverse_file,
    )

    valid_conditions = [
        "faces",
        "objects",
        "fonts",
        "false_fonts",
    ]

    if condition not in valid_conditions:
        raise ValueError(
            f"Condition must be one of {valid_conditions}"
        )

    print(f"\nSelecting {condition} epochs...")

    # Se a tua coluna se chamar "category",
    # muda apenas "Category" para "category"
    epochs_cond = epochs[
        epochs.metadata["category"] == condition
    ]

    print(f"{len(epochs_cond)} epochs selected.")

    print("Computing source estimates...")

    stcs = mne.minimum_norm.apply_inverse_epochs(
        epochs_cond,
        inverse,
        lambda2=1.0 / 9.0,
        method=method,
        pick_ori=None,
        return_generator=False,
    )

    print("Averaging...")

    mean_data = np.mean(
        [stc.data for stc in stcs],
        axis=0,
    )

    mean_stc = stcs[0].copy()
    mean_stc.data = mean_data

    print("Opening interactive brain...")

    brain = mean_stc.plot(
        subject=subject,
        subjects_dir=subjects_dir,
        hemi="both",
        surface="inflated",
        cortex="low_contrast",
        views="lat",
        time_viewer=True,
        clim="auto",
    )

    return brain


if __name__ == "__main__":

    subject = "CA107"

    paths = create_output_folders(subject)

    epochs_file = (
        paths["phase1_epochs"]
        / f"{subject}_04_epochs_meg_Phase1_epo.fif"
    )

    inverse_file = (
        paths["freesurfer"]
        / subject
        / "bem"
        / f"{subject}-meg-inv.fif"
    )

    brain = explore_average_stc(
        subject,
        epochs_file,
        inverse_file,
        paths,
        condition="fonts",      # faces, objects, fonts, false_fonts
        method="eLORETA",
    )
# %%
