import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

st.title("AI Image Detector: DFT Analysis & Separate Blur Analysis")

# Allow a wide range of image file extensions
uploaded_file = st.file_uploader(
    "Upload an image (PNG, JPG, JPEG, BMP, WEBP)",
    type=["png", "jpg", "jpeg", "bmp", "webp"]
)

def compute_dft_features_advanced(gray_image: np.ndarray):
    """
    Compute the 2D DFT magnitude spectrum and extract advanced statistical features:
      - Overall mean and standard deviation.
      - Low vs. high frequency energy ratio.
      - Angular symmetry.
      - Frequency entropy.
      - Periodicity analysis via autocorrelation.
    """
    # 1. Compute 2D DFT and shift to center the zero frequency.
    dft = np.fft.fft2(gray_image)
    dft_shift = np.fft.fftshift(dft)
    mag_spectrum = 20 * np.log(np.abs(dft_shift) + 1)
    
    features = {}
    features["mean_magnitude"] = np.mean(mag_spectrum)
    features["std_magnitude"] = np.std(mag_spectrum)
    
    # 2. Compute low vs. high frequency energy ratio.
    h, w = mag_spectrum.shape
    center_h, center_w = h // 2, w // 2
    radius = min(center_h, center_w) // 4
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center_w)**2 + (Y - center_h)**2)
    mask = dist_from_center <= radius
    low_freq_energy = np.mean(mag_spectrum[mask])
    high_freq_energy = np.mean(mag_spectrum[~mask])
    features["low_freq_energy"] = low_freq_energy
    features["high_freq_energy"] = high_freq_energy
    features["freq_energy_ratio"] = low_freq_energy / (high_freq_energy + 1e-6)
    
    # 3. Angular Symmetry.
    y_indices, x_indices = np.indices(mag_spectrum.shape)
    angles = np.arctan2(y_indices - center_h, x_indices - center_w)
    angles_deg = np.degrees(angles)
    angles_deg[angles_deg < 0] += 360
    bins = np.linspace(0, 360, 361)
    angular_hist, _ = np.histogram(angles_deg.ravel(), bins=bins, weights=mag_spectrum.ravel())
    angular_hist_norm = angular_hist / (np.sum(angular_hist) + 1e-6)
    
    # Compare first 180° with reversed second 180°.
    first_half = angular_hist_norm[:180]
    second_half = angular_hist_norm[180:]
    second_half_reversed = second_half[::-1]
    symmetry_corr = np.corrcoef(first_half, second_half_reversed)[0, 1]
    features["angular_symmetry"] = symmetry_corr
    
    # 4. Frequency Entropy.
    flat_mag = mag_spectrum.ravel()
    flat_mag_norm = flat_mag / (np.sum(flat_mag) + 1e-6)
    freq_entropy = -np.sum(flat_mag_norm * np.log(flat_mag_norm + 1e-6))
    features["frequency_entropy"] = freq_entropy
    
    # 5. Periodicity Analysis via Autocorrelation.
    power_spectrum = np.abs(dft_shift)**2
    auto_corr = np.fft.fftshift(np.abs(np.fft.ifft2(power_spectrum)))
    center_value = auto_corr[center_h, center_w]
    # Zero out a small central region to ignore the main peak.
    auto_corr_copy = np.copy(auto_corr)
    auto_corr_copy[center_h-5:center_h+6, center_w-5:center_w+6] = 0
    second_peak = np.max(auto_corr_copy)
    periodicity_ratio = second_peak / (center_value + 1e-6)
    features["periodicity_ratio"] = periodicity_ratio
    
    return mag_spectrum, features

def compute_blur_measure(gray_image: np.ndarray):
    """
    Compute a blur metric using the variance of the Laplacian.
    A higher variance indicates a sharper image.
    """
    laplacian_var = cv2.Laplacian(gray_image, cv2.CV_64F).var()
    return laplacian_var

if uploaded_file is not None:
    # Read the uploaded file as bytes and open with PIL.
    file_bytes = uploaded_file.read()
    file_stream = io.BytesIO(file_bytes)
    pil_image = Image.open(file_stream).convert("RGB")
    
    # Convert the PIL image to an OpenCV image (BGR) then to grayscale.
    bgr_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    
    # Compute advanced DFT features.
    mag_spectrum, dft_features = compute_dft_features_advanced(gray)
    
    # Compute blur (sharpness) measure.
    blur_measure = compute_blur_measure(gray)
    
    # --- Display the Original Image ---
    st.subheader("Original Image")
    st.image(pil_image, use_column_width=True)
    
    # --- Advanced Frequency (DFT) Analysis ---
    st.subheader("DFT Magnitude Spectrum (Advanced Frequency Analysis)")
    fig, ax = plt.subplots()
    ax.imshow(mag_spectrum, cmap="gray")
    ax.set_title("Magnitude Spectrum")
    ax.axis("off")
    st.pyplot(fig)
    
    st.subheader("Extracted DFT Features")
    st.json(dft_features)
    
    # Heuristic decision for DFT Analysis
    mean_thresh = 90            # Example threshold for mean magnitude
    std_thresh = 55             # Example threshold for standard deviation
    freq_ratio_thresh = 0.9     # Low vs. high frequency energy ratio threshold
    symmetry_thresh = 0.95      # Angular symmetry threshold (lower means more irregular)
    entropy_thresh = 7.0        # Frequency entropy threshold (higher is more natural)
    periodicity_thresh = 0.05   # Periodicity ratio threshold (lower indicates fewer repeating patterns)
    
    conditions_dft = [
        dft_features["mean_magnitude"] < mean_thresh,
        dft_features["std_magnitude"] < std_thresh,
        dft_features["freq_energy_ratio"] < freq_ratio_thresh,
        dft_features["angular_symmetry"] < symmetry_thresh,
        dft_features["frequency_entropy"] > entropy_thresh,
        dft_features["periodicity_ratio"] < periodicity_thresh,
    ]
    dft_condition_count = sum(conditions_dft)
    
    st.write("Number of conditions met in DFT analysis:", dft_condition_count, "out of", len(conditions_dft))
    
    if dft_condition_count >= 4:
        st.success("DFT Analysis: The image appears REAL based on frequency features.")
    else:
        st.error("DFT Analysis: The image shows frequency patterns that may indicate AI-generation.")
    
    # --- Blur Analysis (Separated) ---
    st.subheader("Blur Analysis")
    st.write("Laplacian Variance (Blur Measure):", blur_measure)
    
    # For blur analysis, a lower Laplacian variance indicates more blur, which is often seen in natural images.
    # In contrast, an overly sharp image (high Laplacian variance) might indicate artificial generation.
    sharpness_thresh = 150.0  # Example threshold for sharpness.
    
    if blur_measure < sharpness_thresh:
        st.success("Blur Analysis: The image has a moderate level of blur and appears REAL.")
    else:
        st.error("Blur Analysis: The image is overly sharp, which might indicate AI-generation.")
