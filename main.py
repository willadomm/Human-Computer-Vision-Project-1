import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.transform import SimilarityTransform, warp
import cv2

SIZE_OF_GAUSSIAN_FILTER = 9
SIGMA_OF_GAUSSIAN_FILTER = 2.5
HIGH_PASS_WEIGHT = 1.1

def get_eye_points_opencv(image):
    
    h, w = image.shape[:2]
    detector = cv2.FaceDetectorYN.create("face_detection_yunet_2026may.onnx", "", (w, h))
    _, faces = detector.detect(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    face = faces[0]
    right_eye = face[4:6]
    left_eye = face[6:8]
    return np.array([left_eye, right_eye])


def align_opencv(img1, img2):
    pts1 = get_eye_points_opencv(img1)
    pts2 = get_eye_points_opencv(img2)
    return align_faces(img1, img2, pts1, pts2)

def pick_points(image, n=2, title="Click points"):
    plt.imshow(image)
    plt.title(title)
    pts = plt.ginput(n)
    plt.close()
    return np.array(pts)


def align_faces(img1, img2, pts1=None, pts2=None):
    if pts1 is None:
        pts1 = pick_points(img1, n=2, title="Iamge 1: click left eye, then right eye")
    if pts2 is None:
        pts2 = pick_points(img2, n=2, title="Image 2: click left eye, then right eye")

    transform = SimilarityTransform()
    transform.estimate(pts1, pts2)

    img1_float = img1.astype(np.float64) / 255.0
    aligned1 = warp(img1_float, transform.inverse, output_shape=img2.shape[:2], mode="edge")
    aligned1 = (aligned1 * 255).astype(np.uint8)

    return aligned1


def edge_pad(image, pad_height, pad_width):
    if image.ndim == 3:
        return np.pad(image, ((pad_height, pad_height), (pad_width, pad_width), (0, 0)), mode="edge")
    return np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode="edge")



def gaussian_kernel(size, sigma):
    kernel = np.zeros((size, size))
    center = size // 2

    for i in range(size):
        for j in range(size):
            x, y = i - center, j - center
            kernel[i, j] = (1 / (2 * np.pi * sigma**2)) * np.exp(-(x**2 + y**2) / (2 * sigma**2))

    return kernel / np.sum(kernel)


def turnkernelnegative(kernel):
    Hk, Wk = kernel.shape


def conv_2D(image, kernel):

    Hk, Wk = kernel.shape
    pad_height = Hk // 2
    pad_width = Wk // 2

    if len(image.shape) > 2:
        Hi, Wi, colors = image.shape
        out = np.zeros((Hi, Wi, colors), dtype=np.float64)
        padded_image = edge_pad(image, pad_height, pad_width)

        for c in range(colors):
            for i in range(Hi):
                for j in range(Wi):
                    neighborhood = padded_image[i:i+Hk, j:j+Wk, c]
                    out[i, j, c] = np.sum(neighborhood * kernel)
    else:
        Hi, Wi = image.shape
        out = np.zeros((Hi, Wi), dtype=np.float64)
        padded_image = edge_pad(image, pad_height, pad_width)

        for i in range(Hi):
            for j in range(Wi):
                neighborhood = padded_image[i:i+Hk, j:j+Wk]
                out[i, j] = np.sum(neighborhood * kernel)

    return out


def get_blurredimage(image): 

    blurkernel = gaussian_kernel(SIZE_OF_GAUSSIAN_FILTER, SIGMA_OF_GAUSSIAN_FILTER)

    out = conv_2D(image, blurkernel)

    return out

    

def get_highpassimage(image):
    blurredimage = get_blurredimage(image)
    highpassimage = (image - blurredimage) + 127

    return highpassimage


def combine_images(lowpassimage, highpassimage):
    out = lowpassimage  + highpassimage * HIGH_PASS_WEIGHT
    return out


def main():
    img1 = np.array(Image.open("dataset/medium/face_0008.png").convert("RGB"))
    img2 = np.array(Image.open("dataset/medium/face_0009.png").convert("RGB"))

    print("Choose alignment method:")
    print("  1) No alignment")
    print("  2) Machine Learning")
    print("  3) Manual eye indication")
    choice = input("Enter 1, 2, or 3: ")

    if choice == "1":
        aligned1 = img1
    if choice == "2":
        aligned1 = align_opencv(img1, img2)
    if choice == "3":
        aligned1 = align_faces(img1, img2)

    img1lowpass = get_blurredimage(aligned1)
    img2highpass = get_highpassimage(img2)
    out = combine_images(img1lowpass, img2highpass)
    out = np.clip(out, 0, 255).astype(np.uint8)

    outfinal = Image.fromarray(out)
    outfinal.show()


main()


    
















