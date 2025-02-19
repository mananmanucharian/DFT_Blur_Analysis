import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

st.title("Advanced AI Image Detector")
st.write("This tool performs two separate analyses: Frequency (DFT) Analysis and Blur Analysis.")

# Allow a wide range of image file extensions
uploaded_file = st.file_uploader(
    "Upload an image (PNG, JPG, JPEG, BMP, WEBP)",
    type=["png", "jpg", "jpeg", "bmp", "webp"]
)

def compute_dft_features(gray_image: np.ndarray):
    """
    Compute the 2D DFT of the image and extract key frequency domain features.
    Returns the magnitude spectrum and a dictionary of features.
    """
    # Compute 2D DFT and shift so the low frequencies are at the center.
    dft = np.fft.fft2(gray_image)
    dft_shift = np.fft.fftshift(dft)
    mag_spectrum = 20 * np.log(np.abs(dft_shift) + 1)
    
    features = {}
    features["mean_magnitude"] = np.mean(mag_spectrum)
    features["std_magnitude"] = np.std(mag_spectrum)
    
    # Frequency Energy Ratio (low frequency vs high frequency)
    h, w = mag_spectrum.shape
    center_h, center_w = h // 2, w // 2
    radius = min(center_h, center_w) // 4
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center_w)**2 + (Y - center_h)**2)
    mask = dist_from_center <= radius
    low_freq_energy = np.mean(mag_spectrum[mask])
    high_freq_energy = np.mean(mag_spectrum[~mask])
    features["freq_energy_ratio"] = low_freq_energy / (high_freq_energy + 1e-6)
    
    # Angular Symmetry: Compute an angular histogram and compare two halves.
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
    
    # Frequency Entropy: Measures the randomness in the frequency domain.
    flat_mag = mag_spectrum.ravel()
    flat_mag_norm = flat_mag / (np.sum(flat_mag) + 1e-6)
    freq_entropy = -np.sum(flat_mag_norm * np.log(flat_mag_norm + 1e-6))
    features["frequency_entropy"] = freq_entropy
    
    return mag_spectrum, features

def compute_blur_measure(gray_image: np.ndarray):
    """
    Compute a blur metric using the variance of the Laplacian.
    A higher variance typically means the image is sharper.
    """
    laplacian_var = cv2.Laplacian(gray_image, cv2.CV_64F).var()
    return laplacian_var

if uploaded_file is not None:
    # Load image using PIL (handles many file formats)
    file_bytes = uploaded_file.read()
    file_stream = io.BytesIO(file_bytes)
    pil_image = Image.open(file_stream).convert("RGB")
    
    # Convert PIL image to OpenCV format and then to grayscale.
    bgr_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    
    # Compute Frequency (DFT) Analysis.
    mag_spectrum, dft_features = compute_dft_features(gray)
    
    # Compute Blur Analysis (Laplacian Variance).
    blur_measure = compute_blur_measure(gray)
    
    # Display the original image.
    st.subheader("Original Image")
    st.image(pil_image, use_column_width=True)
    
    # Display the DFT magnitude spectrum.
    fig, ax = plt.subplots()
    ax.imshow(mag_spectrum, cmap="gray")
    ax.set_title("DFT Magnitude Spectrum")
    ax.axis("off")
    st.pyplot(fig)
    
    # Display Frequency Analysis Features.
    st.subheader("Frequency Domain Features (DFT Analysis)")
    st.write("Mean Magnitude: {:.2f}".format(dft_features["mean_magnitude"]))
    st.write("Standard Deviation of Magnitude: {:.2f}".format(dft_features["std_magnitude"]))
    st.write("Frequency Entropy: {:.2f}".format(dft_features["frequency_entropy"]))
    st.write("Frequency Energy Ratio: {:.2f}".format(dft_features["freq_energy_ratio"]))
    st.write("Angular Symmetry: {:.2f}".format(dft_features["angular_symmetry"]))
    
    # Display Blur Analysis.
    st.subheader("Blur Analysis (Laplacian Variance)")
    st.write("Laplacian Variance (Blur Measure): {:.2f}".format(blur_measure))
    
    # --- Frequency Analysis Heuristic ---
    # Here, standard deviation and frequency entropy are the most critical.
    freq_ai = False
    freq_message = ""
    if dft_features["std_magnitude"] < 22.7:
        freq_ai = True
        freq_message += "Standard Deviation of Magnitude is too low (<22.7). "
    if dft_features["frequency_entropy"] < 13.5:
        freq_ai = True
        freq_message += "Frequency Entropy is too low (<13.5). "
    
    if freq_ai:
        st.error("Frequency Analysis indicates AI-generation: " + freq_message)
    else:
        st.success("Frequency Analysis indicates the image appears REAL based on key frequency metrics.")
    
    # --- Blur Analysis Heuristic ---
    # Using a hypothetical threshold for Laplacian variance.
    sharpness_thresh = 150.0  # Adjust this value based on your dataset.
    if blur_measure > sharpness_thresh:
        blur_message = "Image is overly sharp (Laplacian Variance > {:.2f}).".format(sharpness_thresh)
        st.warning("Blur Analysis suggests AI-generation: " + blur_message)
    else:
        blur_message = "Image has normal blur characteristics (Laplacian Variance <= {:.2f}).".format(sharpness_thresh)
        st.info("Blur Analysis suggests the image appears natural: " + blur_message)
    
    # --- Overall Decision ---
    # Since frequency analysis is considered more important, it drives the overall decision.
    overall_decision = "REAL"
    if freq_ai:
        overall_decision = "AI-generated"
    
    st.subheader("Overall Decision")
    st.write("Based on frequency analysis (priority) and separate blur analysis, the image is likely: **{}**".format(overall_decision))
