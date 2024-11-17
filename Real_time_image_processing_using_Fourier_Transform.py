import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to apply Fourier Transform to the image
def apply_fourier_transform(frame):
    # Convert to grayscale for processing
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Fast Fourier Transform (FFT)
    f_transform = np.fft.fft2(gray_frame)

    # Shift the zero frequency component (DC component) to the center
    f_shift = np.fft.fftshift(f_transform)

    return f_shift

# Function to apply filters (low-pass or high-pass) in the frequency domain
def apply_filter(f_shift, filter_type='low-pass', radius=30):
    rows, cols = f_shift.shape
    crow, ccol = rows // 2, cols // 2

    # Create a mask with the same size as the FFT output
    mask = np.zeros_like(f_shift)

    if filter_type == 'low-pass':
        # Low-pass filter: Pass low frequencies, block high frequencies
        mask[crow-radius:crow+radius, ccol-radius:ccol+radius] = 1
    elif filter_type == 'high-pass':
        # High-pass filter: Block low frequencies, pass high frequencies
        mask[:crow-radius, :] = 1
        mask[crow+radius:, :] = 1
        mask[:, :ccol-radius] = 1
        mask[:, ccol+radius:] = 1

    # Apply the mask (filter) on the frequency domain representation
    f_shift_filtered = f_shift * mask

    return f_shift_filtered

# Function to apply Inverse Fourier Transform to convert back to spatial domain
def inverse_fourier_transform(f_shift_filtered):
    # Perform the inverse shift to move the zero frequency component back
    f_ishift = np.fft.ifftshift(f_shift_filtered)

    # Apply Inverse FFT to convert the image back to the spatial domain
    img_back = np.fft.ifft2(f_ishift)

    # Take only the real part of the result (ignore the imaginary part)
    img_back = np.abs(img_back)

    return img_back

# Function to normalize and scale the image for display purposes
def normalize_image(image):
    # Normalize the image to a range of 0 to 255
    norm_image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

    # Convert the image to 8-bit for display
    norm_image = np.uint8(norm_image)

    return norm_image

# Main function to perform real-time video capture and processing
def main():
    # Start video capture from the default webcam
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()  # Capture a frame from the webcam
        if not ret:
            break

        # Apply Fourier Transform to the frame
        f_shift = apply_fourier_transform(frame)

        # Apply a filter (low-pass or high-pass)
        filter_type = 'low-pass'  # Options: 'low-pass' or 'high-pass'
        f_shift_filtered = apply_filter(f_shift, filter_type=filter_type, radius=50)

        # Convert the filtered result back to the spatial domain
        img_back = inverse_fourier_transform(f_shift_filtered)

        # Normalize the result for display
        filtered_image = normalize_image(img_back)

        # Display the original frame and the filtered frame side by side
        combined_frame = np.hstack((frame, cv2.cvtColor(filtered_image, cv2.COLOR_GRAY2BGR)))

        # Display the combined output
        cv2.imshow('Original (Left) | Filtered (Right)', combined_frame)

        # Check if 'q' is pressed or window is closed
        if cv2.getWindowProperty('Original (Left) | Filtered (Right)', cv2.WND_PROP_VISIBLE) < 1:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

# Generate sample data for the graph (filter radius vs. frequency response)
filter_radius = np.linspace(0, 100, 100)
response = 60 * np.exp(-filter_radius / 20)  # Example response curve

# Create the plot
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(filter_radius, response, marker='*', color='b')

# Set axis labels and title
ax.set_xlabel('Filter Radius', fontsize=12)
ax.set_ylabel('Frequency Response', fontsize=12)
ax.set_title('Frequency Response vs. Filter Radius', fontsize=14)

ax.grid(True)
plt.show()