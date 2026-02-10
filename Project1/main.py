import cv2
import numpy as np
from pathlib import Path

# Im not entirely sure if these paths are what they are looking for or not. The directions say that we dont need to zip up the data
# but to ensure the code calls the data files by the same names as the zip file. I am pretty sure this is what they mean but I am not sure.
BASE_DIR = Path(__file__).parent
OFFICE_IMAGES = BASE_DIR / "Office"
REDCHAIR_IMAGES = BASE_DIR / "RedChair"
ENTEREXIT_IMAGES = BASE_DIR / "EnterExitCrossingPaths2cor"

def read_frames(image_dir):
    """Reads in a squence of image frames. #1 in the breakdown of tasks"""
    frames = []
    for image_file in image_dir.glob("*.jpg"):
        # I think all of these images are already in order so we dont have to sort anything
        frame = cv2.imread(str(image_file))
        grayscale_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(grayscale_image)

    return frames

def temporal_derivative(frames, kernel):
    """Applies a 1D differential operator and returns the temporal derivate. #2 in the breakdown of tasks"""
    derivatives = []

    # I am assuming that when the instructions say "temporal derivative" it just means this filter is applies to 
    # time adjacent frames instead of convoluting the pixels in the image (I very likely could be wrong tho)
    for i in range(1, len(frames)-1):
        d_frame = sum([kernel[0]*frames[i-1], kernel[1]*frames[i], kernel[2]*frames[i+1]])
        derivatives.append(d_frame)

    return derivatives

def threshold_frames(frames, threshold=0):
    """Thresholds the frames and creates a mask of 0 or 1s. #3 in teh breakdown of tasks"""
    # should create bitmasks of 0s and 1s based on threshold
    return [(np.abs(frame) > threshold).astype(np.uint8) for frame in frames]

def apply_masks(frames, masks):
    """Applies the appropriate masks in the form of a bitwise AND. I am not really sure if this is how he wants the masks applied, but its how I did them in a class I took in undergrad"""
    output = []
    for frame, mask in zip(frames, masks):
        result = frame.copy()
        # NOTE: this basically converts the picture to black and white but imo it is easier to see the motion this way
        # we could get rid of the result[mask == 1] = 255 to keep the motion in the regular grayscale values idrc
        result[mask == 0] = 0
        result[mask == 1] = 255 
        output.append(result)

    return output

def box_filter(frames,kernel):
    sframes=[]
    for frame in frames:
        blur=cv2.filter2D(frame,-1,kernel)
        sframes.append(blur)
    return sframes

def gaussian(frames, std, size=(0,0)):
    sframes=[]
    for frame in frames:
        blur=cv2.GaussianBlur(frame,size,std)
        sframes.append(blur)
    return  sframes

def derivative_of_guassian(frames, temporal_kernel, t_sigma, threshold=0, size=(3,3)):
    """Derivative of Guassian filter. #1 in the different variations. (I think this is what it is asking for)"""
    # apply guassian to smooth first then take the temporal derivative
    smoothed_frames = gaussian(frames, t_sigma, size)
    d_frames = temporal_derivative(smoothed_frames, temporal_kernel)    
    masks = threshold_frames(d_frames, threshold)
    motion_frames = apply_masks(frames, masks)
    return motion_frames, masks

def main():
    std=2.5
    box_filter_size=5
    kernel2 = np.ones((box_filter_size,box_filter_size),np.float32)/(box_filter_size**2)

    frames = read_frames(OFFICE_IMAGES)
    #frames = box_filter(frames,kernel2)
    # frames = gaussian(frames, std)
    
    temporal_kernel = 0.5 * np.array([-1, 0, 1])
    t_sigma = 0.15 # idk
    threshold = 5 # idk
    size = (3, 3)
    output, masks = derivative_of_guassian(frames, temporal_kernel, t_sigma, threshold, size)

    for frame, mask, out in zip(frames, masks, output):
        combined = np.hstack((frame, mask, out))
        cv2.imshow("original frames, mask, motion frames", combined)
        key = cv2.waitKey(10) # 10 ms
        if key == ord("q"):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()