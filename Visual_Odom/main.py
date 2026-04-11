import cv2
from pathlib import Path

KITTI_IMAGES = None
EUROC_IMAGES = None

def read_frames(image_dir):
    frames = []
    for image_file in image_dir:
        frame = cv2.imread(str(image_file))
        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(grayscale)
    return frames

def main():
    pass

if __name__ == "__main__":
    pass