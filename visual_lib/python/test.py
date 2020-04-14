# This Python file uses the following encoding: utf-8
from argparse import ArgumentParser
import torch
import cv2
import pyvilib
from pyvilib import Frame, FastDetectorCpuGrid, FastDetectorGpu
import numpy as np
from time import time


def main(use_gpu=False):
    print(torch.__version__)
    print(dir(pyvilib))

    parser = ArgumentParser(description='VILIB tester')
    parser.add_argument('source', type=str, help='Source video file path')
    args = parser.parse_args()
    source = args.source
    cap = cv2.VideoCapture(args.source)
    detector = None
    tt, frame_time, detect_time, count = time(), 0.0, 0.0, 0
    while True:
        flag, image = cap.read()
        if not flag:
            break
        if detector is None:
            h, w, c = image.shape
            detector = (
                FastDetectorGpu if use_gpu else FastDetectorCpuGrid
            )(w, h, 64, 64, 0, 1, 0, 0, 10.0, 10, pyvilib.SUM_OF_ABS_DIFF_ON_ARC)

        image = torch.tensor(image[:, :, :1])
        t1 = time()
        frame = Frame(image, 0, 2)
        t2 = time()
        detector.reset()
        detector.detect(frame.pyramid_gpu() if use_gpu else frame.pyramid_cpu())
        t3 = time()
        count += 1
        frame_time += t2 - t1
        detect_time += t3 - t2
        if t3 > tt + 5:
            print(f'Frame time {frame_time / count * 1000}ms, detect time {detect_time / count * 1000}ms')
            tt, frame_time, detect_time, count = time(), 0.0, 0.0, 0
        detector.display_features('Features', frame.pyramid_cpu(), True)
        k = cv2.waitKey(1)
        if k == ord('q'):
            break

    # points_cpu = cpu.get_points()
    # print('CPU found points:', len(points_cpu))
    # cpu.display_features('CPU features', frame.pyramid_cpu(), True)

    # gpu.reset()
    # gpu.detect(frame.pyramid_gpu())
    # points_gpu = gpu.get_points()
    # print('GPU found points:', len(points_gpu))
    # gpu.display_features('GPU features', frame.pyramid_cpu(), True)


if __name__ == "__main__":
    main()
