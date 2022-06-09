import os
import shutil
import argparse
from pathlib import Path
from glob import glob

import cv2
from tqdm import tqdm

SRC_EXTENSION = ".png"
REF_EXTENSION = ".png"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--source",
    "-t",
    type=str,
    default="./source/",
    help="Path to segmentation label (source) dir",
)
parser.add_argument(
    "--reference",
    "-r",
    type=str,
    default="./reference/",
    help="Path to reference dir",
)
parser.add_argument(
    "--out",
    "-o",
    type=str,
    default="./label_new_out/",
    help="Path to output dir for new label",
)
args = parser.parse_args()

src_path = args.source
ref_path = args.reference
out_path = args.out


def main():
    src_files = sorted(glob(os.path.join(src_path, "*" + SRC_EXTENSION)))
    ref_files = sorted(glob(os.path.join(ref_path, "*" + REF_EXTENSION)))

    if len(src_files) != len(ref_files):
        print(
            "[Warning] Number of files is different. "
            + "Non-existent images are ignored."
        )
        print("Src. :" + str(len(src_files)))
        print("Ref. :" + str(len(ref_files)))
    else:
        print("Number of images: " + str(len(src_files)))

    Path(out_path).mkdir(parents=True, exist_ok=True)

    for num in tqdm(range(len(src_files))):
        src_filename, _ = os.path.splitext(os.path.basename(src_files[num]))
        ref_file = os.path.join(ref_path, src_filename + REF_EXTENSION)
        src_img = cv2.imread(src_files[num])
        ref_img = cv2.imread(ref_file)

        if src_img is None:
            print(
                "[Info] Src image is not found, skip it: "
                + str(src_files[num])
            )
            continue
        elif ref_img is None:
            print("[Info] Ref image is not found, skip it: " + str(ref_file))
            continue

        src_height, src_width = src_img.shape[:2]
        ref_height, ref_width = ref_img.shape[:2]

        dst_path = os.path.join(out_path, src_filename + SRC_EXTENSION)

        if src_height != ref_height or src_width != ref_width:
            out_img = cv2.resize(
                src_img,
                (ref_width, ref_height),
                interpolation=cv2.INTER_NEAREST,
            )
            cv2.imwrite(dst_path, out_img)
        else:
            shutil.copy(
                os.path.join(src_path, src_filename + SRC_EXTENSION), dst_path
            )


if __name__ == "__main__":
    main()
