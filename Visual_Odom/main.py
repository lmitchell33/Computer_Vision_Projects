import cv2
from pathlib import Path
import numpy as np
import time

"""
I think there are a couple of things to note/keep track of about about the KITTI dataset. Firstly, there are 22 total image/calibration 
sequences (labeled 00-21), however, there are only 11 ground truth datasets (labeled 00-10). I think this
has something to do with like a test/train split. Anyways, this means that we should only use numbers 00-10
for testing since we want to have the ability to show the ground truth as well as our result. Also note that
within each sequence there are 2 directories of images, "image_0" which is the left camera and "image_1" which 
is the right camera.

Another thing, the car they used had two sets of cameras one grayscale and another color, in the calibration data/files they give us 
there are 4 sets of flattened calibration matricies. They never specify in the readme in the devkit, but I think that the the first two 
(P0 and P1) are the grayscale matricies. In the devkit they also state that all images have been undistorted and rectified, meaning 
that the intrinsic matrix and epipolar stuff has already been applied to the images

The truth directory has the ground truth data which has a 12 column table containing the pose of the LEFT camera at the current timestep. 
The time intervals/steps are in text files in both the image and calibration directories. They also note that the left cameras coord
system has the Z-axis facing forward, as you would expect. Each row is a 3x4 transformation matrix (in ground truth poses)
"""

# NOTE: images/calibration each have an extra layer of directories compared to ground truth not really sure why but thats how the data came 
KITTI_IMAGES_BASE_DIR = Path(__file__).parent / "KITTI/images/sequences"
# KITTI_CALIBRATION_BASE_DIR = Path(__file__).parent + "KITTI/calibration/sequences"
KITTI_GROUND_TRUTH_BASE_DIR = Path(__file__).parent / "KITTI/truth/poses"


def read_frames(image_dir):
    frames = []
    for image_file in image_dir.glob("*.png"):
        frame = cv2.imread(str(image_file), cv2.IMREAD_GRAYSCALE)
        frames.append(frame)
    return frames

def get_features(image, algorithm="orb", threshold_scale=0.01):
    """NOTE: just change the alg arg to use a different feature detection alg"""
    # NOTE: this page is awesome for this: https://docs.opencv.org/4.x/db/d27/tutorial_py_table_of_contents_feature2d.html

    curr_time = time.time()

    if algorithm == "orb":
        orb = cv2.ORB.create()
        keypoints = orb.detect(image)
        keypoints, descriptor = orb.compute(image, keypoints)

    elif algorithm == "harris":
        # the ORB and sift detectors both output a KeyPoint object so just for
        # consistency, it would be easier if the harris also returns the same obj
        response_map = cv2.cornerHarris(np.float32(image), 3, 3, 0.04)
        # response_map = cv2.dilate(response_map, None) 
        corner_mask = response_map > (threshold_scale*response_map.max())
        corner_coords_y, corner_coords_x = np.where(corner_mask == True)
        # print(corner_coords_x, corner_coords_y)
        keypoints = [cv2.KeyPoint(float(x), float(y), 3) for y, x in zip(corner_coords_y, corner_coords_x)]
        descriptor = None

    elif algorithm == "sift":
        sift = cv2.SIFT.create()
        keypoints = sift.detect(image, None)
        keypoints, descriptor = sift.compute(image, keypoints)

    elif algorithm == "surf":
        # lol apparently surf is patented so we need another lib. It doesnt look too hard
        # if he wanted to do it, but idk if its necessary
        pass 
    else:
        raise Exception("Invalid algorithm argument")
    
    time_diff = time.time() - curr_time
    return keypoints, descriptor, time_diff

def match_features(prev_frame, curr_Frame):
    # also include outlier removal here
    pass

def estimate_pose():
    pass

def visual_odometry(left_images, right_images, time, ground_truth):
    feature_detection_time = []
    feature_algorithm = "orb"

    # now that im thinking about it im not really sure if we even need to do this stuff
    # for both left and right frames. Thats a later me problem
    l_prev_features, l_prev_descriptor, l_time = get_features(left_images[0], feature_algorithm)
    r_prev_features, r_prev_descriptor, r_time = get_features(right_images[0], feature_algorithm)
    feature_detection_time.append(np.mean([l_time, r_time]))
    for i in range(1, len(left_images)):
        l_frame = left_images[i]
        r_frame = right_images[i]
        l_features, l_descriptor, l_time = get_features(l_frame, feature_algorithm)
        r_features, r_descriptor, r_time = get_features(r_frame, feature_algorithm)
        feature_detection_time.append(np.mean([l_time, r_time]))

    print(f"Average time to detect features for {feature_algorithm}: {round(np.mean(feature_detection_time), 5)} seconds")

def main():
    suequence_num = "00"
    kitti_left_images = KITTI_IMAGES_BASE_DIR / f"{suequence_num}/image_0"
    kitti_right_images = KITTI_IMAGES_BASE_DIR / f"{suequence_num}/image_1"

    kitti_calibration = KITTI_IMAGES_BASE_DIR / f"{suequence_num}/calib.txt"
    KITTI_time = KITTI_IMAGES_BASE_DIR / f"{suequence_num}/times.txt"

    kitti_ground_truth = KITTI_GROUND_TRUTH_BASE_DIR / f"{suequence_num}.txt"

    left_frames = read_frames(kitti_left_images)
    # right_frames = read_frames(kitti_right_images)

    feature_detection_time = []
    feature_algorithm = "orb"
    for frame in left_frames:
        features, descriptor, time_diff = get_features(frame, algorithm=feature_algorithm)
        feature_detection_time.append(time_diff)

        featured_image = cv2.drawKeypoints(frame, features, None)
        combined_images = np.hstack((cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR), featured_image))
        cv2.imshow("base image and base image with keypoints", combined_images)
        key = cv2.waitKey(30)
        if key == ord("q"):
            break

    print(f"Average time to detect features for {feature_algorithm}: {round(np.mean(feature_detection_time), 5)} seconds")

    # visual_odometry(left_frames, right_frames)

if __name__ == "__main__":
    main()