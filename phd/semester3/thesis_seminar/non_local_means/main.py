import time
import numpy as np
import matplotlib.pyplot as plt

from skimage import io, data, img_as_float
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.metrics import peak_signal_noise_ratio
from skimage.util import random_noise

def print_data(name, image):
    print(name, image.shape, image.ndim, image.dtype, np.min(image), np.max(image))
#astro = img_as_float(data.astronaut())
#astro = astro[30:180, 150:300]
astro = img_as_float(io.imread('S1A_IW_GRDH_1SDV_20170608T100350_20170608T100415_016941_01C325_4013.tif'))
astro = np.expand_dims(astro, axis=-1)
astro = astro.astype(np.float64)
print_data("original", astro)

sigma = 0.08
noisy = random_noise(astro, var=sigma**2)
print_data("noisy", noisy)

# estimate the noise standard deviation from the noisy image
sigma_est = np.mean(estimate_sigma(noisy, channel_axis=-1))
print(f'estimated noise standard deviation = {sigma_est}')

patch_kw = dict(patch_size=5,      # 5x5 patches
                patch_distance=6,  # 13x13 search area
                channel_axis=-1)

# slow algorithm
tic = time.perf_counter()
denoise = denoise_nl_means(noisy, h=1.15 * sigma_est, fast_mode=False,
                           **patch_kw)
denoise = np.expand_dims(denoise, axis=-1)
print_data("denoise", denoise)
toc = time.perf_counter()
print(f"Slow algorithm in {toc - tic:0.4f} seconds")

# slow algorithm, sigma provided
tic = time.perf_counter()
denoise2 = denoise_nl_means(noisy, h=0.8 * sigma_est, sigma=sigma_est,
                            fast_mode=False, **patch_kw)
denoise2 = np.expand_dims(denoise2, axis=-1)
print_data("denoise2", denoise2)
toc = time.perf_counter()
print(f"Slow algorithm, sigma provided in {toc - tic:0.4f} seconds")

# fast algorithm
tic = time.perf_counter()
denoise_fast = denoise_nl_means(noisy, h=0.8 * sigma_est, fast_mode=True,
                                **patch_kw)
denoise_fast = np.expand_dims(denoise_fast, axis=-1)
print_data("denoise_fast", denoise_fast)
toc = time.perf_counter()
print(f"Fast algorithm in {toc - tic:0.4f} seconds")

# fast algorithm, sigma provided
tic = time.perf_counter()
denoise2_fast = denoise_nl_means(noisy, h=0.6 * sigma_est, sigma=sigma_est,
                                 fast_mode=True, **patch_kw)
denoise2_fast = np.expand_dims(denoise2_fast, axis=-1)
print_data("denoise2_fast", denoise2_fast)
toc = time.perf_counter()
print(f"Fast algorithm, sigma provided in {toc - tic:0.4f} seconds")

fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(8, 6),
                       sharex=True, sharey=True)

ax[0, 0].imshow(noisy)
ax[0, 0].axis('off')
ax[0, 0].set_title('noisy')
ax[0, 1].imshow(denoise)
ax[0, 1].axis('off')
ax[0, 1].set_title('non-local means\n(slow)')
ax[0, 2].imshow(denoise2)
ax[0, 2].axis('off')
ax[0, 2].set_title('non-local means\n(slow, using $\\sigma_{est}$)')
ax[1, 0].imshow(astro)
ax[1, 0].axis('off')
ax[1, 0].set_title('original\n(noise free)')
ax[1, 1].imshow(denoise_fast)
ax[1, 1].axis('off')
ax[1, 1].set_title('non-local means\n(fast)')
ax[1, 2].imshow(denoise2_fast)
ax[1, 2].axis('off')
ax[1, 2].set_title('non-local means\n(fast, using $\\sigma_{est}$)')

fig.tight_layout()
plt.savefig("figure1.png")

# print PSNR metric for each case
img_min = np.min(astro)
img_max = np.max(astro)
img_range = img_max - img_min
psnr_noisy = peak_signal_noise_ratio(astro, noisy, data_range=img_range)
psnr = peak_signal_noise_ratio(astro, denoise, data_range=img_range)
psnr2 = peak_signal_noise_ratio(astro, denoise2, data_range=img_range)
psnr_fast = peak_signal_noise_ratio(astro, denoise_fast, data_range=img_range)
psnr2_fast = peak_signal_noise_ratio(astro, denoise2_fast, data_range=img_range)

print(f'PSNR (noisy) = {psnr_noisy:0.2f}')
print(f'PSNR (slow) = {psnr:0.2f}')
print(f'PSNR (slow, using sigma) = {psnr2:0.2f}')
print(f'PSNR (fast) = {psnr_fast:0.2f}')
print(f'PSNR (fast, using sigma) = {psnr2_fast:0.2f}')

#plt.show()
