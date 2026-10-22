import numpy as np
import argparse
from PIL import Image
import matplotlib.pyplot as plt

SIZE_OF_GAUSSIAN_FILTER = 11
SIGMA_OF_GAUSSIAN_FILTER = 5
HIGH_PASS_WEIGHT = 2



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
    highpassimage = (image - blurredimage)

    return highpassimage


def combine_images(lowpassimage, highpassimage):
    out = lowpassimage  + (highpassimage * HIGH_PASS_WEIGHT)
    return out


#Method One


def to_gray(image):
    if image.ndim == 3:
        return image.mean(axis=2)
    return image.astype(np.float64)


def estimate_translation_ssd(image_a, image_b, max_shift=20, sigma=2.0):
    
    size = 2 * int(3 * sigma) + 1
    kernel = gaussian_kernel(size, sigma)
    a = conv_2D(to_gray(image_a), kernel)
    b = conv_2D(to_gray(image_b), kernel)
    H, W = a.shape
    m = int(3 * sigma)  

    best_error = np.inf
    best_shift = (0, 0)
    for v in range(-max_shift, max_shift + 1):
        for u in range(-max_shift, max_shift + 1):
            
            y0, y1 = max(0, -v) + m, min(H, H - v) - m
            x0, x1 = max(0, -u) + m, min(W, W - u) - m
            patch_a = a[y0:y1, x0:x1]
            patch_b = b[y0 + v:y1 + v, x0 + u:x1 + u]
            patch_a = patch_a - patch_a.mean()   
            patch_b = patch_b - patch_b.mean()
            error = np.mean((patch_a - patch_b) ** 2)   
            if error < best_error:
                best_error = error
                best_shift = (u, v)

    return best_shift


#Method 2


def realign_image(image, dx, dy):
    
    H, W = image.shape[0], image.shape[1]
    out = np.zeros(image.shape, dtype=np.float64)
    y0, y1 = max(0, -dy), min(H, H - dy)
    x0, x1 = max(0, -dx), min(W, W - dx)
    out[y0:y1, x0:x1] = image[y0 + dy:y1 + dy, x0 + dx:x1 + dx]
    return out


def sobel_gradients(gray):
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=np.float64)
    sobel_y = sobel_x.T
    return conv_2D(gray, sobel_x), conv_2D(gray, sobel_y)


def harris_response(gray, sigma=1.5, k=0.05):
    Ix, Iy = sobel_gradients(gray)

    
    kernel = gaussian_kernel(2 * int(3 * sigma) + 1, sigma)
    A = conv_2D(Ix * Ix, kernel)
    B = conv_2D(Ix * Iy, kernel)
    C = conv_2D(Iy * Iy, kernel)

    det = A * C - B ** 2
    trace = A + C
    return det - k * trace ** 2          


def harris_keypoints(gray, num_points=200, border=20, nms_radius=5, threshold_ratio=0.01):
    
    R = harris_response(gray)
    H, W = R.shape
    if H <= 2 * border or W <= 2 * border:
        return np.zeros((0, 2), dtype=int)    
    threshold = threshold_ratio * R[border:H - border, border:W - border].max()
 
    points = []
    for y, x in np.argwhere(R > threshold):
        if x < border or x >= W - border or y < border or y >= H - border:
            continue                          
        window = R[y - nms_radius:y + nms_radius + 1, x - nms_radius:x + nms_radius + 1]
        if R[y, x] == window.max():          
            points.append((R[y, x], x, y))
 
    points.sort(reverse=True)
    return np.array([(x, y) for _, x, y in points[:num_points]]).reshape(-1, 2)


def get_patch(gray, x, y, radius):
    patch = gray[y - radius:y + radius + 1, x - radius:x + radius + 1].ravel()
    patch = patch - patch.mean()
    return patch / (patch.std() + 1e-8)


def vote_for_translation(displacements, tolerance=1):
    best_count = 0
    best_shift = (0, 0)
    for d in displacements:
        close = np.all(np.abs(displacements - d) <= tolerance, axis=1)
        if close.sum() > best_count:
            best_count = close.sum()
            best_shift = np.median(displacements[close], axis=0)
    return int(round(best_shift[0])), int(round(best_shift[1]))


def estimate_translation_harris(image_a, image_b, max_shift=20, patch_radius=7, num_points=200):
    a = to_gray(image_a)
    b = to_gray(image_b)

    border = max_shift + patch_radius
    points_a = harris_keypoints(a, num_points, border)
    points_b = harris_keypoints(b, num_points, border)
    if len(points_a) == 0 or len(points_b) == 0:
        return (0, 0)

    patches_a = np.array([get_patch(a, x, y, patch_radius) for x, y in points_a])
    patches_b = np.array([get_patch(b, x, y, patch_radius) for x, y in points_b])

   
    displacements = []
    for i in range(len(points_a)):
        ssd = np.sum((patches_b - patches_a[i]) ** 2, axis=1)
        j = np.argmin(ssd)
        displacements.append(points_b[j] - points_a[i])  
    displacements = np.array(displacements)

    
    displacements = displacements[np.all(np.abs(displacements) <= max_shift, axis=1)]
    if len(displacements) == 0:
        return (0, 0)

    return vote_for_translation(displacements)

def load_image(path):
    return np.array(Image.open(path).convert("RGB"), dtype=np.float64)
 
 
def main(argv=None):    
    parser = argparse.ArgumentParser(description="Make a hybrid image, optionally aligning the two images first.")
    parser.add_argument("image_a", help="image whose low frequencies are used")
    parser.add_argument("image_b", help="image whose high frequencies are used (assumed to be the shifted one)")
    parser.add_argument("--method", choices=["none", "ssd", "harris"], default="none",
                        help="how to estimate the shift before combining: none (no alignment), "
                             "ssd (Method A), or harris (Method B). Default: none")
    parser.add_argument("--max-shift", type=int, default=20, help="largest shift (in pixels) to look for")
    parser.add_argument("--output", default="hybrid.png", help="where to save the hybrid image")
    parser.add_argument("--no-show", action="store_true", help="save the result without opening a window")
    args = parser.parse_args(argv)
 
    img_a = load_image(args.image_a)
    img_b = load_image(args.image_b)
    if img_a.shape != img_b.shape:
        parser.error("the two images must be the same size")
 
     
    if args.method == "ssd":
        dx, dy = estimate_translation_ssd(img_a, img_b, max_shift=args.max_shift)
    elif args.method == "harris":
        dx, dy = estimate_translation_harris(img_a, img_b, max_shift=args.max_shift)
    else:
        dx, dy = 0, 0
    print(f"method: {args.method}   estimated shift of image_b: dx = {dx}, dy = {dy}")
 
    if (dx, dy) != (0, 0):
        img_b = realign_image(img_b, dx, dy)
 
    
    low = get_blurredimage(img_a)
    high = get_highpassimage(img_b)
    hybrid = combine_images(low, high)
    hybrid = np.clip(hybrid, 0, 255).astype(np.uint8)
 
    Image.fromarray(hybrid).save(args.output)
    print(f"saved {args.output}")
    if not args.no_show:
        plt.imshow(hybrid)
        plt.axis("off")
        plt.show()
 
 
if __name__ == "__main__":
    main()