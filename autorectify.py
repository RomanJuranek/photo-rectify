"""
Automatic photo rectification

(c) 2020, Roman Juranek <ijuranek@fit.vutbr.cz>

Development of this software was funded by TACR project TH04010394,
Progressive Image Processing Algorithms.
"""


import argparse
import logging
import pathlib
import skimage

import lgroup


def parse_args():
    pass


if __name__ == "__main__":
    args = parse_args()

    for filename in args.files:
        logging.info(f"Processing {filename}")
        image = skimage.io.imread(filename)

