from qPAI_cINN_uncertainty_estimation.data import load_spectra_file
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter


# Phantom1_flow_phantom_medium_melanin
# Phantom2_flow_phantom_medium_melanin
DATASET = "Phantom1_flow_phantom_medium_melanin"
path_name = fr"I:\research\seblab\data\group_folders\Janek\learned_pa_oximetry/" + \
    fr"validation_data\in_vitro\{DATASET}\{DATASET}.npz"

(wavelengths, oxygenations, lu_values, spectra, melanin_concentration,
    background_oxygenation, distances, depths, timesteps, tumour_mask, reference_mask,
    mouse_body_mask, background_mask) = load_spectra_file(path_name)

distinct_timesteps = np.unique(timesteps)
lu = np.asarray([np.mean(lu_values[timesteps == step_value]) for step_value in distinct_timesteps])
lu_std = np.asarray([np.std(lu_values[timesteps == step_value]) for step_value in distinct_timesteps])
pO2_avg = np.asarray([np.mean(oxygenations[timesteps == step_value]) for step_value in distinct_timesteps])
pO2_std = np.asarray([np.std(oxygenations[timesteps == step_value]) for step_value in distinct_timesteps])

plt.figure(figsize=(8, 4))
plt.suptitle(DATASET[9:].replace("_", " "))
plt.subplot(1, 2, 1)
plt.plot(distinct_timesteps, pO2_avg * 100, label="pO$_2$ reference", color="green")
plt.fill_between(distinct_timesteps, (pO2_avg - pO2_std) * 100, (pO2_avg + pO2_std) * 100, color="green", alpha=0.3)
plt.plot(distinct_timesteps, lu * 100, label="linear unmixing", c="blue")
plt.fill_between(distinct_timesteps, (lu - lu_std) * 100, (lu + lu_std) * 100, color="blue", alpha=0.3)
plt.plot(distinct_timesteps, lu * 100, label="learned unmixing", c="orange")
plt.fill_between(distinct_timesteps, (lu - lu_std) * 100, (lu + lu_std) * 100, color="orange", alpha=0.3)
plt.xlabel("Time [s]")
plt.ylabel("Blood oxygenation sO$_2$ [%]")
plt.legend()

plt.subplot(1, 2, 2)
plt.ylim(-35, 35)
plt.hlines(0, xmin=np.min(distinct_timesteps), xmax=np.max(distinct_timesteps), colors="black", linestyles="dashed")
plt.plot(distinct_timesteps, (pO2_avg - lu) * 100, label="linear unmixing", c="blue")
plt.fill_between(distinct_timesteps, (pO2_avg - (lu - lu_std)) * 100, (pO2_avg - (lu + lu_std)) * 100, color="blue", alpha=0.3)
plt.plot(distinct_timesteps, (pO2_avg - lu) * 100, label="learned unmixing", c="orange")
plt.fill_between(distinct_timesteps, (pO2_avg - (lu - lu_std)) * 100, (pO2_avg - (lu + lu_std)) * 100, color="orange", alpha=0.3)
plt.xlabel("Time [s]")
plt.ylabel("Difference from pO$_2$ reference [p.p.]")
plt.legend()
plt.tight_layout()
plt.savefig("../figures/"+DATASET[9:]+".svg")
plt.show()
plt.close()
