import numpy as np
from PIL import Image

SIZE_OF_GAUSSIAN_FILTER = 9
SIGMA_OF_GAUSSIAN_FILTER = 2.5
HIGH_PASS_WEIGHT = .8


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
    img1 = np.array(Image.open("dataset/aligned/face_0002.png").convert("RGB"))
    img1lowpass = get_blurredimage(img1)


    img2 = np.array(Image.open("dataset/aligned/face_0003.png").convert("RGB"))
    img2highpass = get_highpassimage(img2)


    out = combine_images(img1lowpass, img2highpass)
    out = np.clip(out, 0, 255).astype(np.uint8)

    outfinal = Image.fromarray(out)


    outfinal.show()
    


main()













