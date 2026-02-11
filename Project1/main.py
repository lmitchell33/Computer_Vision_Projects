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

    # basically just zero pad the temporal derivative
    d_frame_1 = (kernel[1]*frames[0] - kernel[2]*frames[1])
    derivatives.append(d_frame_1)

    # I am assuming that when the instructions say "temporal derivative" it just means this filter is applies to 
    # time adjacent frames instead of convoluting the pixels in the image (I very likely could be wrong tho)
    for i in range(1, len(frames)-1):
        d_frame = (kernel[0]*frames[i-1] + kernel[1]*frames[i] + kernel[2]*frames[i+1])
        derivatives.append(d_frame)

    # end with zero padding again
    d_frame_end = (kernel[0]*frames[-2] + kernel[1]*frames[-1])
    derivatives.append(d_frame_end)

    return derivatives

def threshold_frames(frames, threshold=0):
    """Thresholds the frames and creates a mask of 0 or 1s. #3 in teh breakdown of tasks"""
    # should create bitmasks of 0s and 1s based on threshold
    return [(np.abs(frame) > threshold).astype(np.uint8) for frame in frames]

def apply_masks(frames, masks):
    """Applies the appropriate masks in the form of a bitwise AND. I am not really sure if this is how he wants the masks applied, but its how I did them in a class I took in undergrad"""
    output = []
    for frame, mask in zip(frames, masks):
        result = frame.copy() # you have add this line or else it will apply the mask to the original images stored in frames

        # NOTE: this basically converts the picture to black and white but imo it is easier to see the motion this way
        # we could get rid of the result[mask == 1] = 255 to keep the motion in the regular grayscale values idrc
        result[mask == 0] = 0
        result[mask == 1] = 255 
        output.append(result)

    return output

def box_filter(frames, size):
    sframes=[]
    box_filter_size = size[0]
    kernel = np.ones((box_filter_size,box_filter_size),np.float32)/(box_filter_size**2)
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

def smooth_then_temporal(frames, temporal_kernel, threshold, s_sigma=0, use_box=True, size=(3,3)):
    """#2 in the different variations. (I think this is what it is asking for)"""
    # apply guassian to smooth first then take the temporal derivative
    if use_box: 
        smoothed_frames = box_filter(frames, size)  
    else:
        smoothed_frames = gaussian(frames, s_sigma, size)

    d_frames = temporal_derivative(smoothed_frames, temporal_kernel)
    masks = threshold_frames(d_frames, threshold)
    motion_frames = apply_masks(frames, masks)
    return motion_frames

def main():
    #frames = box_filter(frames,kernel2)
    # frames = gaussian(frames, std)
    
    frames = read_frames(OFFICE_IMAGES)
    temporal_kernel = 0.5 * np.array([-1, 0, 1])
    s_sigmas = [0.10, 1, 5] # idk
    threshold = 10 # idk
    sizes = [(3,3), (5,5)]
    box_outputs =  [smooth_then_temporal(frames=frames, temporal_kernel=temporal_kernel, threshold=threshold, use_box=True, size=size) for size in sizes]
    guass_outputs = [smooth_then_temporal(frames=frames, temporal_kernel=temporal_kernel, threshold=threshold, s_sigma=s_sigma, use_box=False, size=(3,3)) for s_sigma in s_sigmas]

    for index in range(len(frames)):
        frame = frames[index]
        box_output_1 = box_outputs[0][index]
        box_output_2 = box_outputs[1][index]
        gauss_0 = guass_outputs[0][index]
        gauss_1 = guass_outputs[1][index]
        gauss_2 = guass_outputs[2][index]
        
        combined = np.hstack((frame, box_output_1, box_output_2, gauss_0, gauss_1, gauss_2))
        cv2.imshow(f"Original, box_size={sizes[0]}, box_size={sizes[1]} s_sigma={s_sigmas[0]}, s_sigma={s_sigmas[1]}, s_sigma={s_sigmas[2]}", combined)
        key = cv2.waitKey(30)
        if key == ord("q"):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()