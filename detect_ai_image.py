import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

st.title("AI Image Detector: Combined DFT & Blur Analysis")

# Allow a wide range of image file extensions.
uploaded_file = st.file_uploader(
    "Upload an image (PNG, JPG, JPEG, BMP, WEBP)",
    type=["png", "jpg", "jpeg", "bmp", "webp"]
)

def compute_dft_features(gray_image: np.ndarray):
    """
    Compute the 2D DFT magnitude spectrum and extract advanced statistical features.
    Returns the magnitude spectrum and a dictionary of features.
    """
    # Compute the 2D DFT and shift to center the zero frequency.
    dft = np.fft.fft2(gray_image)
    dft_shift = np.fft.fftshift(dft)
    mag_spectrum = 20 * np.log(np.abs(dft_shift) + 1)
    
    features = {}
    features["mean_magnitude"] = np.mean(mag_spectrum)
    features["std_magnitude"] = np.std(mag_spectrum)
    
    # Compute low vs. high frequency energy ratio.
    h, w = mag_spectrum.shape
    center_h, center_w = h // 2, w // 2
    radius = min(center_h, center_w) // 4
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center_w)**2 + (Y - center_h)**2)
    mask = dist_from_center <= radius
    low_freq_energy = np.mean(mag_spectrum[mask])
    high_freq_energy = np.mean(mag_spectrum[~mask])
    features["freq_energy_ratio"] = low_freq_energy / (high_freq_energy + 1e-6)
    
    # Angular symmetry: compare the angular distribution in two halves.
    y_indices, x_indices = np.indices(mag_spectrum.shape)
    angles = np.arctan2(y_indices - center_h, x_indices - center_w)
    angles_deg = np.degrees(angles)
    angles_deg[angles_deg < 0] += 360
    bins = np.linspace(0, 360, 361)
    angular_hist, _ = np.histogram(angles_deg.ravel(), bins=bins, weights=mag_spectrum.ravel())
    angular_hist_norm = angular_hist / (np.sum(angular_hist) + 1e-6)
    first_half = angular_hist_norm[:180]
    second_half = angular_hist_norm[180:]
    second_half_reversed = second_half[::-1]
    symmetry_corr = np.corrcoef(first_half, second_half_reversed)[0, 1]
    features["angular_symmetry"] = symmetry_corr
    
    # Frequency entropy.
    flat_mag = mag_spectrum.ravel()
    flat_mag_norm = flat_mag / (np.sum(flat_mag) + 1e-6)
    freq_entropy = -np.sum(flat_mag_norm * np.log(flat_mag_norm + 1e-6))
    features["frequency_entropy"] = freq_entropy
    
    # Periodicity analysis via autocorrelation.
    power_spectrum = np.abs(dft_shift)**2
    auto_corr = np.fft.fftshift(np.abs(np.fft.ifft2(power_spectrum)))
    center_value = auto_corr[center_h, center_w]
    auto_corr_copy = np.copy(auto_corr)
    auto_corr_copy[center_h-5:center_h+6, center_w-5:center_w+6] = 0  # Ignore the main peak.
    second_peak = np.max(auto_corr_copy)
    features["periodicity_ratio"] = second_peak / (center_value + 1e-6)
    
    return mag_spectrum, features

def compute_blur_measure(gray_image: np.ndarray):
    """
    Compute the blur metric using the variance of the Laplacian.
    A higher variance indicates a sharper image.
    """
    return cv2.Laplacian(gray_image, cv2.CV_64F).var()

if uploaded_file is not None:
    # Read the uploaded file and open it via PIL.
    file_bytes = uploaded_file.read()
    file_stream = io.BytesIO(file_bytes)
    pil_image = Image.open(file_stream).convert("RGB")
    
    # Convert the PIL image to OpenCV format (BGR) and then to grayscale.
    bgr_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    
    # --- DFT Analysis ---
    mag_spectrum, dft_features = compute_dft_features(gray)
    
    # Display the magnitude spectrum.
    st.subheader("DFT Magnitude Spectrum")
    fig, ax = plt.subplots()
    ax.imshow(mag_spectrum, cmap="gray")
    ax.set_title("Magnitude Spectrum")
    ax.axis("off")
    st.pyplot(fig)
    
    st.subheader("Extracted DFT Features")
    st.json(dft_features)
    
    # *** DFT Decision: The std magnitude is the most important factor.
    # If std magnitude > 23.7 then we classify DFT analysis as REAL.
    if dft_features["std_magnitude"] > 23.7:
        dft_result = "REAL"
    else:
        dft_result = "AI-generated"
    
    st.write("DFT Analysis Decision:", dft_result)
    
    # --- Blur Analysis ---
    blur_measure = compute_blur_measure(gray)
    st.subheader("Blur Analysis")
    st.write("Laplacian Variance (Blur Measure):", blur_measure)
    
    # For blur analysis: we consider the image REAL if it is not overly sharp.
    # (i.e. a lower laplacian variance indicates some natural blur).
    # Here, we use an example threshold: if blur_measure < 150, then it's REAL.
    if blur_measure < 150.0:
        blur_result = "REAL"
    else:
        blur_result = "AI-generated"
    
    st.write("Blur Analysis Decision:", blur_result)
    
    # --- Combined Decision ---
    # Overall, if either analysis (DFT OR Blur) returns REAL, we classify the image as REAL.
    if dft_result == "REAL" or blur_result == "REAL":
        overall_decision = "REAL"
    else:
        overall_decision = "AI-generated"
    
    st.subheader("Overall Decision")
    st.write("Based on the combined DFT and Blur analysis, the image appears:", overall_decision)
