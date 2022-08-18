from qPAI_cINN_uncertainty_estimation.data import load_spectra_file
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt
import numpy as np

PATH = "InVivo1_GasChallengeMouse"

path_name = r"I:\research\seblab\data\group_folders\Janek\learned_pa_oximetry/" + \
    fr"validation_data\in_vivo\{PATH}\{PATH}.npz"

(wavelengths, oxygenations, lu, spectra, melanin_concentration,
    background_oxygenation, distances, depths, timesteps, tumour_mask, reference_mask,
    mouse_body_mask, background_mask) = load_spectra_file(path_name)

num_baseline_averages = 20

distance_transform_tumour = distance_transform_edt(tumour_mask)
distance_transform_reference = distance_transform_edt(reference_mask)

baseline_sO2 = np.mean(oxygenations[:num_baseline_averages, 0], axis=0)

delta_sO2_mean_tumour = [np.mean(oxygenations[i, 0][tumour_mask] - baseline_sO2[tumour_mask]) * 100
                         for i in range(num_baseline_averages, len(oxygenations))]
delta_sO2_rim_tumour = [np.mean(oxygenations[i, 0][tumour_mask & (distance_transform_tumour < 10)] - baseline_sO2[tumour_mask & (distance_transform_tumour < 10)]) * 100
                        for i in range(num_baseline_averages, len(oxygenations))]
delta_sO2_core_tumour = [np.mean(oxygenations[i, 0][tumour_mask & (distance_transform_tumour >= 10)] - baseline_sO2[tumour_mask & (distance_transform_tumour >= 10)]) * 100
                         for i in range(num_baseline_averages, len(oxygenations))]

delta_sO2_mean_reference = [np.mean(oxygenations[i, 0][reference_mask] - baseline_sO2[reference_mask]) * 100
                            for i in range(num_baseline_averages, len(oxygenations))]
delta_sO2_rim_reference = [np.mean(oxygenations[i, 0][reference_mask & (distance_transform_reference < 10)] - baseline_sO2[reference_mask & (distance_transform_reference < 10)]) * 100
                           for i in range(num_baseline_averages, len(oxygenations))]
delta_sO2_core_reference = [np.mean(oxygenations[i, 0][reference_mask & (distance_transform_reference >= 10)] - baseline_sO2[reference_mask & (distance_transform_reference >= 10)]) * 100
                            for i in range(num_baseline_averages, len(oxygenations))]

timesteps = timesteps[num_baseline_averages:] / 1000000 / 60

plt.figure(figsize=(12, 9))

plt.subplot(2, 2, 1)
plt.title("Tumour rim/core")
plt.imshow(spectra[0, 0, :, :], cmap="gray")
plt.imshow(tumour_mask & (distance_transform_tumour < 10), cmap="jet", alpha=0.3, vmax=2)
plt.imshow(tumour_mask & (distance_transform_tumour >= 10), cmap="jet", alpha=0.3)
plt.axis("off")

plt.subplot(2, 2, 2)
plt.title("Reference rim/core")
plt.imshow(spectra[0, 0, :, :], cmap="gray")
plt.imshow(reference_mask & (distance_transform_reference < 10), cmap="jet", alpha=0.3, vmax=2)
plt.imshow(reference_mask & (distance_transform_reference >= 10), cmap="jet", alpha=0.3)
plt.axis("off")

plt.subplot(2, 2, 3)
plt.title("Tumour")
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.plot(timesteps, delta_sO2_mean_tumour, label="mean")
plt.plot(timesteps, delta_sO2_rim_tumour, label="rim")
plt.plot(timesteps, delta_sO2_core_tumour, label="core")
plt.xlabel("Time [min]")
plt.ylabel("$\\delta$ sO$_2$ tumour [p.p.]")
plt.hlines(0, xmin=np.min(timesteps), xmax=np.max(timesteps))
plt.legend()

plt.subplot(2, 2, 4)
plt.title("Reference")
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.plot(timesteps, delta_sO2_mean_reference, label="mean")
plt.plot(timesteps, delta_sO2_rim_reference, label="rim")
plt.plot(timesteps, delta_sO2_core_reference, label="core")
plt.xlabel("Time [min]")
plt.ylabel("$\\delta$ sO$_2$ reference [p.p.]")
plt.hlines(0, xmin=np.min(timesteps), xmax=np.max(timesteps))
plt.legend()

plt.tight_layout()
plt.savefig(f"../figures/{PATH}.svg")
plt.show()
plt.close()
