import cv2
from pathlib import Path
import numpy as np
import time
import matplotlib.pyplot as plt

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

KITTI_IMAGES_BASE_DIR = Path(__file__).parent / "KITTI/images/sequences"
KITTI_GROUND_TRUTH_BASE_DIR = Path(__file__).parent / "KITTI/truth/poses"

def read_frames(image_dir):
    frames = []
    for image_file in image_dir.glob("*.png"):
        frame = cv2.imread(str(image_file), cv2.IMREAD_GRAYSCALE)
        frames.append(frame)
    return frames

def load_times(time_file):
    with open(time_file) as file:
        return [float(line.strip()) for line in file]

def load_ground_truth(ground_truth_file):
    # 3x4 matrices
    with open(ground_truth_file) as file:
        poses = []
        for line in file:
            values = line.strip().split(" ")
            row = [list(map(float, values[i:i+4])) for i in range(0, len(values), 4)]
            poses.append(row)
        
    return np.array(poses)

def load_calibration(calibration_file):
    # 3x4 matrices
    with open(calibration_file) as file:
        lines = file.readlines()

    p0 = lines[0].removeprefix("P0: ").strip().split(" ")
    p1 = lines[1].removeprefix("P1: ").strip().split(" ")
    left_mat = []
    right_mat = []
    for i in range(0, len(p0), 4):
        left_row = list(map(float, p0[i:i+4]))
        right_row = list(map(float, p1[i:i+4]))
        left_mat.append(left_row)
        right_mat.append(right_row)

    #3x4 projection matrices for triangulation
    proj_left = np.array(p0, dtype=np.float64).reshape(3, 4)
    proj_right = np.array(p1, dtype=np.float64).reshape(3, 4)

    # intrinsic matrices = the 3x3 top left box. Base line calculation from here: https://medium.com/@jaimin-k/exploring-kitti-visual-ododmetry-dataset-8ac588246cdc
    left_intrinsic = np.array(left_mat)[0:3, 0:3]
    right_intrinsic = np.array(right_mat)[0:3, 0:3]
    baseline_distance = (left_mat[0][3] - right_mat[0][3]) / left_mat[0][0]

    return left_intrinsic, right_intrinsic, abs(baseline_distance),proj_left,proj_right

def plot_poses(ground_truth_poses, estimated_poses):
    x_gt_data = [p[0][3] for p in ground_truth_poses]
    # y_gt_data = [p[1][3] for p in ground_truth_poses]
    z_gt_data = [p[2][3] for p in ground_truth_poses]

    x_est_data = [p[0][3] for p in estimated_poses]
    #y_est_data = [p[1][3] for p in estimated_poses]
    z_est_data = [p[2][3] for p in estimated_poses]

    # turns out, when plotting the ground truth, you have to plot the x-z data not the x-y data (I tried and it
    # just does not match the video recording). I believe it is because the coord sys for the point gray flea 2 
    # video cameras at this URL: https://www.cvlibs.net/datasets/kitti/setup.php has the y axis pointing down
    # and the X-Z axes pointing outwards with the Z-axis pointing forward and the x axis pointing left-right
    plt.figure()
    plt.plot(x_gt_data, z_gt_data, label="ground truth", color="b")
    plt.plot(x_est_data, z_est_data, label="VO estimation", color="r")
    plt.scatter(x_gt_data[0], z_gt_data[0], color="green", marker="o", s=100, label='start')
    plt.scatter(x_gt_data[-1], z_gt_data[-1], color="red", marker="x", s=100, label='GT end')
    plt.scatter(x_est_data[-1], z_est_data[-1], color="green", marker="x", s=100, label='estimate end')
    plt.legend()
    plt.xlabel("x (m)")
    plt.ylabel("z (m)")
    plt.title("Trajectories")
    plt.tight_layout()
    plt.show()

def display_features(frame, features):
    cv2.namedWindow("features", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("features", 1600, 400)
    featured_image = cv2.drawKeypoints(frame, features, None)
    combined_images = np.hstack((cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR), featured_image))
    cv2.imshow("features", combined_images)
    key = cv2.waitKey(10) # they used 10 hz to 0.1 second = 100 ms (I want to go faster bc it takes too long for testing)
    return key == ord('q')

def display_matches(prev_frame, prev_features, curr_frame, curr_features, matches):
    cv2.namedWindow("matches", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("matches", 1600, 400)
    matched_image = cv2.drawMatches(prev_frame, prev_features, curr_frame, curr_features, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow("matches", matched_image)
    key = cv2.waitKey(10)
    return key == ord("q")

def get_features(image, algorithm="orb", threshold_scale=0.01):
    # NOTE: this page is awesome https://docs.opencv.org/4.x/db/d27/tutorial_py_table_of_contents_feature2d.html
    curr_time = time.time()

    if algorithm == "orb":
        orb = cv2.ORB.create(nfeatures=1000)
        keypoints = orb.detect(image)
        time_diff = time.time() - curr_time

        keypoints, descriptor = orb.compute(image, keypoints)

    elif algorithm == "harris":
        # the ORB and sift detectors both output a KeyPoint object so just for
        # consistency, it is easier to convert the Harris output into that same obj
        response_map = cv2.cornerHarris(np.float32(image), 3, 3, 0.04)
        corner_mask = response_map > (threshold_scale*response_map.max())
        corner_coords_y, corner_coords_x = np.where(corner_mask == True)
        time_diff = time.time() - curr_time

        keypoints = [cv2.KeyPoint(float(x), float(y), 3) for y, x in zip(corner_coords_y, corner_coords_x)]
        descriptor = None

    elif algorithm == "sift":
        sift = cv2.SIFT.create()
        keypoints = sift.detect(image, None)
        time_diff = time.time() - curr_time

        keypoints, descriptor = sift.compute(image, keypoints)

    else:
        raise Exception("Invalid algorithm argument")
    
    return keypoints, descriptor, time_diff

def match_features_brute_force(prev_descriptor, curr_descriptor, algorithm="orb"):
    # https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html
    curr_time = time.time()

    if algorithm == "orb":
        bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf_matcher.match(prev_descriptor, curr_descriptor)

    elif algorithm == "sift":
        bf_matcher = cv2.BFMatcher(cv2.NORM_L2)
        matches = bf_matcher.knnMatch(prev_descriptor, curr_descriptor, k=2)

        # ratio test between two nearest neighbors (k=2) with threshold of 0.75
        good = []
        for m, n in matches:
            if m.distance < (0.75 * n.distance):
                good.append(m)

        matches = good
 
    time_diff = time.time() - curr_time

    prev_matched_indicies = [point.queryIdx for point in matches]
    curr_matched_indicies = [point.trainIdx for point in matches]

    return prev_matched_indicies, curr_matched_indicies, matches, time_diff

def match_features_optical_flow(prev_frame, curr_frame, prev_features, ):
    # I guess this would technically be feature tracking but the output would be roughly the same.
    # TODO: this whole thing later or just remove it
    p1, st, err = cv2.calcOpticalFlowPyrLK(prev_frame, curr_frame, prev_features, None)
    pass



def match_stereo_features(left_frame, right_frame, left_keypoints, left_descriptor, algorithm="orb"):
    """Match features between left and right images at the same timestep."""
    right_keypoints, right_descriptor, _ = get_features(right_frame, algorithm)
    _, _, matches, _ = match_features_brute_force(left_descriptor, right_descriptor, algorithm)

    left_pts = np.float64([left_keypoints[m.queryIdx].pt for m in matches])
    right_pts = np.float64([right_keypoints[m.trainIdx].pt for m in matches])
    left_indices = [m.queryIdx for m in matches]

    return left_pts, right_pts, left_indices

def common_matched_features(prev_indices, curr_indices,curr_features,stereo_left_pts,stereo_right_pts, stereo_left_indices):
        temporal_map = {p: c for p, c in zip(prev_indices, curr_indices)}
        common_stereo_pts_left = []
        common_stereo_pts_right = []
        common_curr_pts = []

        for idx, stereo_idx in enumerate(stereo_left_indices):
            if stereo_idx in temporal_map:
                common_stereo_pts_left.append(stereo_left_pts[idx])
                common_stereo_pts_right.append(stereo_right_pts[idx])
                curr_feat_idx = temporal_map[stereo_idx]
                common_curr_pts.append(curr_features[curr_feat_idx].pt)

        common_stereo_pts_left = np.float64(common_stereo_pts_left)
        common_stereo_pts_right = np.float64(common_stereo_pts_right)
        common_curr_pts = np.float64(common_curr_pts)
        return common_stereo_pts_left,common_stereo_pts_right,common_curr_pts


def triangulate_points(left_pts, right_pts, proj_left, proj_right):
    """Triangulate 3D points from stereo correspondences."""
    # cv2.triangulatePoints expects 2xN arrays
    pts_4d = cv2.triangulatePoints(proj_left, proj_right, left_pts.T, right_pts.T)

    # Convert from homogeneous (4D) to 3D by dividing by w
    pts_3d = (pts_4d[:3] / pts_4d[3]).T  # Nx3

    # Filter out points with negative depth or very far away points
    valid = (pts_3d[:, 2] > 0) & (pts_3d[:, 2] < 200)
    return pts_3d, valid


def estimate_pose_pnp(pts_3d, pts_2d, intrinsic_matrix):
    """Estimate camera pose from 3D-2D correspondences using PnP + RANSAC."""
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        pts_3d, pts_2d, intrinsic_matrix, 
        distCoeffs=None,
        iterationsCount=500,
        reprojectionError=1.0,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return None, None

    R, _ = cv2.Rodrigues(rvec)  # convert rotation vector to matrix
    t = tvec.flatten()

    return R, t

def visual_odometry(left_images, right_images, left_intrinsic, proj_left, proj_right, ground_truth_poses, feature_algorithm="orb"):
    feature_detection_time = []
    feature_count = []

    match_time = []
    match_count = []

    current_pose = np.eye(4)
    estimated_poses = [current_pose[:3, :].copy()]

    prev_features, prev_descriptor, detect_time = get_features(left_images[0], feature_algorithm)
    prev_frame = left_images[0]
    feature_detection_time.append(detect_time)
    feature_count.append(len(prev_features))

    for i in range(1, len(left_images)):
        curr_frame = left_images[i]

        curr_features, curr_descriptor, detect_time = get_features(curr_frame, feature_algorithm)
        feature_detection_time.append(detect_time)
        feature_count.append(len(curr_features))
        exit = display_features(curr_frame, curr_features)
        if exit: 
            break

        prev_indicies, curr_indicies, matches, match_time_diff = match_features_brute_force( prev_descriptor, curr_descriptor, feature_algorithm)
        # match_features_optical_flow(prev_frame, curr_frame, prev_features)
        match_time.append(match_time_diff)
        match_count.append(len(matches))
        exit = display_matches(prev_frame, prev_features, curr_frame, curr_features, matches[:30])
        if exit:
            break
        
        # Match left[k-1] <-> right[k-1]     
        stereo_left_pts, stereo_right_pts, stereo_left_indices = match_stereo_features(prev_frame, right_images[i - 1], prev_features, prev_descriptor, feature_algorithm)

        #Find features with both temporal and spatial matches
        #stereo_set = set(stereo_left_indices)
        common_stereo_pts_left,common_stereo_pts_right,common_curr_pts=common_matched_features(prev_indicies, curr_indicies,curr_features,stereo_left_pts,stereo_right_pts, stereo_left_indices)

        #Get 3D Points
        pts_3d, valid = triangulate_points(common_stereo_pts_left, common_stereo_pts_right, proj_left, proj_right)

        R, t = estimate_pose_pnp(pts_3d, common_curr_pts, left_intrinsic)

        if R is not None:
            # PnP gives us the transform FROM world TO camera, so the world frame is the inverse
            T = np.eye(4)
            T[:3, :3] = R.T
            T[:3, 3] = -R.T @ t
            current_pose = current_pose @ T
        
        estimated_poses.append(current_pose[:3, :].copy())


        # update
        prev_features = curr_features
        prev_descriptor = curr_descriptor
        prev_frame = curr_frame

    # plot the ground truth with estimated trajectory and print out any stats here
    plot_poses(ground_truth_poses, estimated_poses)
    print("Feature detection stats: ")
    print(f"Average time to detect features for {feature_algorithm}: {round(np.mean(feature_detection_time), 5)*1000} ms")
    print(f"Average number of features detected for {feature_algorithm}: {int(np.mean(feature_count))}")

    print(" ")
    print("Feature matching stats: ")
    print(f"Average time to match features for {feature_algorithm}: {round(np.mean(match_time), 5)*1000} ms")
    print(f"Average number of matches for {feature_algorithm}: {int(np.mean(match_count))}")

def main():
    # 00, 03, 05, 06, 07, 08, 09, and 10 all have fairly stationary scenes
    # 01, 02, and 04 are all on busier roads/highways and have more dynamic scenes
    sequence_num = "00"
    kitti_left_images = KITTI_IMAGES_BASE_DIR / f"{sequence_num}/image_0"
    kitti_right_images = KITTI_IMAGES_BASE_DIR / f"{sequence_num}/image_1"

    kitti_calibration = KITTI_IMAGES_BASE_DIR / f"{sequence_num}/calib.txt"
    left_intrinsic, right_intrinsic, baseline, proj_left, proj_right = load_calibration(kitti_calibration)
    print(f"Left intrinsic matrix for sequences {sequence_num}: \n {left_intrinsic} \n")
    print(f"Right intrinsic matrix for sequence {sequence_num}: \n {right_intrinsic} \n")
    print(f"Baseline between left and right camera: {baseline}")
    print(" ")

    kitti_ground_truth = KITTI_GROUND_TRUTH_BASE_DIR / f"{sequence_num}.txt"
    ground_truth = load_ground_truth(kitti_ground_truth)

    left_frames = read_frames(kitti_left_images)
    right_frames = read_frames(kitti_right_images)

    feature_algorithm = "orb"

    visual_odometry(
        left_images=left_frames, 
        right_images=right_frames, 
        left_intrinsic=left_intrinsic, 
        proj_left=proj_left, 
        proj_right=proj_right,
        ground_truth_poses=ground_truth, 
        feature_algorithm=feature_algorithm
    )

if __name__ == "__main__":
    main()