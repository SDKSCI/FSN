import numpy as np
import matplotlib.pyplot as plt
from skimage import io, segmentation, color
from skimage.segmentation import slic
from skimage.segmentation import quickshift  # skimage没有直接LSC，这里用quickshift代替示范
import cv2
from scipy.stats import mode
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
def superpixel_downsample(img, label_map, output_size):
    H, W, C = img.shape
    H_out, W_out = output_size

    h_scale = H / H_out
    w_scale = W / W_out

    img_ds = np.zeros((H_out, W_out, C), dtype=img.dtype)
    label_ds = np.zeros((H_out, W_out), dtype=label_map.dtype)

    for i_out in range(H_out):
        for j_out in range(W_out):
            h_start = int(np.floor(i_out * h_scale))
            h_end = int(np.ceil((i_out + 1) * h_scale))
            w_start = int(np.floor(j_out * w_scale))
            w_end = int(np.ceil((j_out + 1) * w_scale))

            h_end = min(h_end, H)
            w_end = min(w_end, W)

            region_labels = label_map[h_start:h_end, w_start:w_end].flatten()
            region_pixels = img[h_start:h_end, w_start:w_end, :].reshape(-1, C)

            if len(region_labels) == 0:
                continue

            sp_label = mode(region_labels, axis=None)[0][0]
            label_ds[i_out, j_out] = sp_label

            mask = (region_labels == sp_label)
            sp_pixels = region_pixels[mask]

            if len(sp_pixels) == 0:
                pixel_val = np.mean(region_pixels, axis=0)
            else:
                pixel_val = np.mean(sp_pixels, axis=0)

            img_ds[i_out, j_out, :] = pixel_val

    return img_ds, label_ds

def main():
    # 读取图片
    img = io.imread(r"G:\Data\BSDS500\Clean\images\10081.jpg")
    img = img.astype(np.float32) / 255.

    H_out, W_out =1000, 1000

    # 1) SLIC 超像素
    slic_labels = slic(img, n_segments=1000, compactness=10, sigma=1, start_label=0)

    # 2) LSC 超像素
    # skimage没有内置LSC算法，通常用OpenCV实现
    # 这里用OpenCV的ximgproc模块的 createSuperpixelLSC
    import cv2.ximgproc as ximgproc
    img_bgr = (img * 255).astype(np.uint8)[..., ::-1]  # RGB->BGR

    slic_lsc = ximgproc.createSuperpixelLSC(img_bgr, region_size=10, ratio=0.075)
    slic_lsc.iterate(10)
    mask_lsc = slic_lsc.getLabelContourMask(False)
    lsc_labels = slic_lsc.getLabels()

    print(f'LSC 超像素数量: {slic_lsc.getNumberOfSuperpixels()}')

    # 普通下采样（双线性插值）
    img_bilinear = cv2.resize(img_bgr, (W_out, H_out), interpolation=cv2.INTER_LINEAR)
    img_bilinear_rgb = img_bilinear[..., ::-1] / 255.

    # 用无损下采样函数分别对SLIC和LSC标签下采样
    img_slic_ds, _ = superpixel_downsample(img, slic_labels, (H_out, W_out))
    img_lsc_ds, _ = superpixel_downsample(img, lsc_labels, (H_out, W_out))


    fig, axs = plt.subplots(2, 4, figsize=(16, 8))

    # ---- 第一行 SLIC ----
    axs[0, 0].imshow(img)
    axs[0, 0].set_title('原始图像')
    axs[0, 0].axis('off')

    axs[0, 1].imshow(segmentation.mark_boundaries(img, slic_labels, color=(1, 0, 0)))
    axs[0, 1].set_title('SLIC 超像素')
    axs[0, 1].axis('off')

    axs[0, 2].imshow(img_slic_ds)
    axs[0, 2].set_title('基于 SLIC 下采样')
    axs[0, 2].axis('off')

    axs[0, 3].imshow(img_bilinear_rgb)
    axs[0, 3].set_title('普通双线性下采样')
    axs[0, 3].axis('off')

    # ---- 第二行 LSC ----
    axs[1, 0].imshow(img)
    axs[1, 0].set_title('原始图像')
    axs[1, 0].axis('off')

    axs[1, 1].imshow(segmentation.mark_boundaries(img, lsc_labels, color=(0, 1, 0)))
    axs[1, 1].set_title('LSC 超像素')
    axs[1, 1].axis('off')

    axs[1, 2].imshow(img_lsc_ds)
    axs[1, 2].set_title('基于 LSC 下采样')
    axs[1, 2].axis('off')

    axs[1, 3].imshow(img_bilinear_rgb)
    axs[1, 3].set_title('普通双线性下采样')
    axs[1, 3].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
