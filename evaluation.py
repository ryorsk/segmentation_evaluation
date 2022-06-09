import os
import sys
import time
import argparse
from glob import glob

import numpy as np
import cv2
import numba as nb
from numba import prange

import color_labels

# PLEASE CHANGE ##############################################
COLOR_TYPE = "CLASS"
N_CLASS = 28
GT_EXTENSION = ".png"
RES_EXTENSION = ".png"
# END ########################################################

parser = argparse.ArgumentParser()
parser.add_argument(
    "--teach",
    "-t",
    type=str,
    default="./label/",
    help="Path to segmentation label dir",
)
parser.add_argument(
    "--result",
    "-r",
    type=str,
    default="./results/",
    help="Path to results dir",
)
args = parser.parse_args()

gt_path = args.teach
result_path = args.result


@nb.jit(parallel=True)
def calc_matrix(matrix, index, gt_img, res_img, color_array):
    img_height, img_width = gt_img.shape[:2]
    for height in prange(img_height):
        for width in prange(img_width):
            for input in prange(index):
                if (
                    gt_img[height, width, 2] == color_array[input, 0]
                    and gt_img[height, width, 1] == color_array[input, 1]
                    and gt_img[height, width, 0] == color_array[input, 2]
                ):
                    for output in prange(index):
                        if (
                            res_img[height, width, 2] == color_array[output, 0]
                            and res_img[height, width, 1]
                            == color_array[output, 1]
                            and res_img[height, width, 0]
                            == color_array[output, 2]
                        ):
                            matrix[input, output] += 1.0
    return matrix


@nb.jit(parallel=True)
def extract_pixel_color(color_array, gt_img, res_img, img_height, img_width):
    index = 0
    b = gt_img[:, :, 0]
    g = gt_img[:, :, 1]
    r = gt_img[:, :, 2]

    for height in prange(img_height):
        for width in prange(img_width):
            b = gt_img[height, width, 0]
            g = gt_img[height, width, 1]
            r = gt_img[height, width, 2]

            same = 0
            for counter in prange(index):
                if (
                    color_array[counter, 0] == r
                    and color_array[counter, 1] == g
                    and color_array[counter, 2] == b
                ):
                    same += 1

            if same == 0:
                np.vstack((color_array, (r, g, b)))
                color_array[index, 0] = r
                color_array[index, 1] = g
                color_array[index, 2] = b
                index += 1

    return index, color_array


def get_image_size(gt_img, res_img, file_name):
    if (
        gt_img.shape[0] != res_img.shape[0]
        or gt_img.shape[1] != res_img.shape[1]
    ):
        print("[Error] Image size is different:" + file_name)
        sys.exit()
    else:
        img_height, img_width = gt_img.shape[:2]
        return img_height, img_width


def extract_all_colors(gt_files, res_files):
    color_array = np.ones([900, 3])
    color_array[:, :] = 999
    num_gt_files = len(gt_files)
    print("Extracting RGB Color")
    for num in range(num_gt_files):
        gt_filename, gt_ext = os.path.splitext(os.path.basename(gt_files[num]))
        res_filename, res_ext = os.path.splitext(
            os.path.basename(res_files[num])
        )
        gt_img = cv2.imread(os.path.join(gt_path, gt_filename + gt_ext))
        res_img = cv2.imread(os.path.join(result_path, res_filename + res_ext))

        img_height, img_width = get_image_size(gt_img, res_img, gt_filename)
        index, color_array = extract_pixel_color(
            color_array, gt_img, res_img, img_height, img_width
        )
        print(str(num + 1) + "/" + str(num_gt_files))

    print("Extracted RGB colors: " + str(index))
    return color_array[:index, :]


def evaluation(matrix, index, gt_files, res_files, color_array):
    num_gt_files = len(gt_files)
    print("Evaluation Start")
    for num in range(num_gt_files):
        gt_filename, gt_ext = os.path.splitext(os.path.basename(gt_files[num]))
        res_filename, res_ext = os.path.splitext(
            os.path.basename(res_files[num])
        )
        gt_img = cv2.imread(
            os.path.join(gt_path, gt_filename + gt_ext)
        ).astype(np.uint16)
        res_img = cv2.imread(
            os.path.join(result_path, res_filename + res_ext)
        ).astype(np.uint16)
        img_height, img_width = get_image_size(gt_img, res_img, gt_filename)

        start_time = time.time()
        matrix = calc_matrix(matrix, index, gt_img, res_img, color_array)
        elapsed = time.time() - start_time
        print(
            str(num + 1)
            + "/"
            + str(num_gt_files)
            + ":"
            + res_filename
            + " calctime:"
            + str(elapsed)
            + "[sec]"
        )

    return matrix


def main():
    gt_files = sorted(glob(os.path.join(gt_path, "*" + GT_EXTENSION)))
    res_files = sorted(glob(os.path.join(result_path, "*" + RES_EXTENSION)))

    if len(gt_files) != len(res_files):
        print("[Error] Each(GT & result) number of files is different")
        print("GT    :" + str(len(gt_files)))
        print("result:" + str(len(res_files)))
        sys.exit()
    else:
        print("Number of result images: " + str(len(gt_files)))

    if COLOR_TYPE == "None":
        color_array = extract_all_colors(gt_files, res_files)
    elif COLOR_TYPE == "ADE":
        color_array = color_labels.ade20k_label_colors
    elif COLOR_TYPE == "CLASS":
        color_array = np.array([[i, i, i] for i in range(0, N_CLASS)])

    index = len(color_array)
    color_array = color_array.astype(np.float64)
    print("Included colors: " + str(index))

    matrix = np.zeros([index, index]).astype(np.float32)
    matrix = evaluation(matrix, index, gt_files, res_files, color_array)

    # Global acc. = True pix / All pix
    G_pixel = 0
    for G_num in range(index):
        G_pixel += matrix[G_num, G_num]
    GA = G_pixel / matrix.sum()
    print("Global Accuracy: " + str(GA * 100) + "%")

    # Class acc. = True pix each class / All pix each class
    ca = 0
    for C_num in range(index):
        if matrix[C_num, :].sum() == 0.0:
            continue
        ca += matrix[C_num, C_num] / matrix[C_num, :].sum()
    CA = ca / index
    print("Class Accuracy : " + str(CA * 100) + "%")

    # Mean IoU
    mi = 0
    for M_num in range(index):
        if (
            matrix[M_num, :].sum()
            + matrix[:, M_num].sum()
            - matrix[M_num, M_num]
            == 0.0
        ):
            continue
        mi += matrix[M_num, M_num] / (
            matrix[M_num, :].sum()
            + matrix[:, M_num].sum()
            - matrix[M_num, M_num]
        )
    MI = mi / index
    print("Mean IoU       : " + str(MI * 100) + "%")

    if np.isnan(GA) or np.isnan(CA) or np.isnan(MI):
        print(
            "[Warning] Some accuracy couldn't be calculated correctly. "
            + "Please set [COLOR_TYPE = None] and use colour extractor. "
            + "This takes a lot of time."
        )

    # file output
    f = open(os.path.join(result_path, "segment_result.txt"), "w")
    f.writelines("Total Result" + "\n")
    f.writelines("Global Accuracy:" + str(GA * 100) + "%" + "\n")
    f.writelines("Class Accuracy:" + str(CA * 100) + "%" + "\n")
    f.writelines("Mean IoU:" + str(MI * 100) + "%" "\n")


if __name__ == "__main__":
    main()
