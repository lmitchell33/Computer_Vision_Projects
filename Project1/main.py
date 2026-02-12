import cv2
import numpy as np
from pathlib import Path
from scipy.ndimage import convolve1d

BASE_DIR = Path(__file__).parent
OFFICE_IMAGES = BASE_DIR / "Office"
REDCHAIR_IMAGES = BASE_DIR / "RedChair"
ENTEREXIT_IMAGES = BASE_DIR / "EnterExitCrossingPaths2cor"

def read_frames(image_dir):
    """Reads in a squence of image frames. #1 in the breakdown of tasks"""
    frames = []
    for image_file in image_dir.glob("*.jpg"):
        # i think all of these images are already in order so we dont have to sort anything
        frame = cv2.imread(str(image_file))
        grayscale_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(grayscale_image)

    return frames

def temporal_derivative(frames, kernel):
    """Applies a 1D differential operator and returns the temporal derivate. #2 in the breakdown of tasks"""
    derivatives = []

    # basically just zero pad the temporal derivative 
    d_frame_1 = (kernel[1]*frames[0] + kernel[2]*frames[1])
    derivatives.append(d_frame_1)

    # assuming that when the instructions say "temporal derivative" it just means this filter is applies to 
    # time adjacent frames instead of convoluting the pixels in the image
    for i in range(1, len(frames)-1):
        d_frame = (kernel[0]*frames[i-1] + kernel[1]*frames[i] + kernel[2]*frames[i+1])
        derivatives.append(d_frame)

    # end with zero padding again
    d_frame_end = (kernel[0]*frames[-2] + kernel[1]*frames[-1])
    derivatives.append(d_frame_end)

    # we purposefully do not type cast to uint8 here because leaving the prevision as floats seems to
    # do an immensely better job at removing the noise in the final motion frames
    
    return derivatives

def threshold_frames(frames, threshold=None):
    """Thresholds the frames and creates a mask of 0 or 1s. #3 in teh breakdown of tasks"""
    # should create "bitmasks" of 0s and 1s based on threshold
    if threshold:
        return [(np.abs(frame) > threshold).astype(np.uint8) for frame in frames]

    return [(np.abs(frame) > find_good_threshold(frame)).astype(np.uint8) for frame in frames]

def apply_masks(frames, masks):
    """Applies the appropriate masks in the form of a bitwise AND. I am not really sure if this is how he wants the masks applied, but its how I did them in a class I took in undergrad"""
    output = []
    for frame, mask in zip(frames, masks):
        result = np.copy(frame) # you have add this line or else it will apply the mask to the original images stored in frames
        # result = result * mask
        result[mask == 0] = 0
        # result[mask == 1] = 255 
        output.append(result)

    return output



def derivative_gaussian(frames, tsigma, threshold=None):
    """Full pipeline of applying 1D Gaussian filter"""
    #Auto calculate kernel size
    ksize = int(6 * tsigma + 1)
    if ksize % 2 == 0:
        ksize += 1
    #Build kernel
    half = ksize // 2
    x = np.arange(-half, half + 1, dtype=np.float64)
    kernel = -x / (tsigma ** 2) * np.exp(-x ** 2 / (2 * tsigma ** 2))
    kernel = kernel / np.sum(np.abs(kernel)) * 2

    #Applies the filter i guess?
    frames_array = np.array(frames, dtype=np.float64).copy()
    result = convolve1d(frames_array, kernel, axis=0, mode='nearest')

    #Apply masking
    masks = threshold_frames(result, threshold)
    motion_frames = apply_masks(frames, masks)  # Pass original list
    return motion_frames



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

def smooth_then_temporal(frames, temporal_kernel, threshold=None, s_sigma=0, use_box=True, size=(3,3)):
    """#2 in the different variations"""
    # apply gaussian to smooth first then take the temporal derivative
    if use_box: 
        smoothed_frames = box_filter(frames, size)  
    else:
        smoothed_frames = gaussian(frames, s_sigma, size)

    d_frames = temporal_derivative(smoothed_frames, temporal_kernel)
    masks = threshold_frames(d_frames, threshold)
    motion_frames = apply_masks(frames, masks)
    return motion_frames



def find_good_threshold(frame):
    """#3 in the different variations"""
    # 1 std away is little low and keeps way too much noise but 2 and 3 both seem decent but 3 is the best in my opionin  
    return 3*np.std(frame)


def main():    
    # frames = read_frames(OFFICE_IMAGES)
    # frames = read_frames(REDCHAIR_IMAGES)
    frames = read_frames(ENTEREXIT_IMAGES)

    temporal_kernel = 0.5 * np.array([-1, 0, 1])
    s_sigmas = [0.30, 2, 10] # idk
    t_sigmas = [0.30, 2, 10]
    thresholds = [5, 25, 50] # idk
    sizes = [(3,3), (5,5)]

    # right now they are all just using the calculated thresholds
    box_outputs =  [smooth_then_temporal(frames=frames, temporal_kernel=temporal_kernel, use_box=True, size=size) for size in sizes]
    guass_outputs = [smooth_then_temporal(frames=frames, temporal_kernel=temporal_kernel, s_sigma=s_sigma, use_box=False, size=(3,3)) for s_sigma in s_sigmas]
    dog_outputs = [derivative_gaussian(frames=frames,tsigma=t_sigma) for t_sigma in t_sigmas]

    for index in range(len(frames)):
        frame = frames[index]
        dog_0 = dog_outputs[0][index]
        dog_1 = dog_outputs[1][index]
        dog_2 = dog_outputs[2][index]
        dog_filter_frames = np.hstack((dog_0, dog_1, dog_2))

        box_output_1 = box_outputs[0][index]
        box_output_2 = box_outputs[1][index]
        box_filter_frames = np.hstack((frame, box_output_1, box_output_2))

        gauss_0 = guass_outputs[0][index]
        gauss_1 = guass_outputs[1][index]
        gauss_2 = guass_outputs[2][index]
        gauss_filter_frames = np.hstack((gauss_0, gauss_1, gauss_2))

        combined = np.vstack((dog_filter_frames, box_filter_frames, gauss_filter_frames))
        cv2.imshow(f"t_sigma={t_sigmas[0]}, t_sigma={t_sigmas[1]}, t_sigma={t_sigmas[2]} | Original, box_size={sizes[0]}, box_size={sizes[1]} | s_sigma={s_sigmas[0]}, s_sigma={s_sigmas[1]}, s_sigma={s_sigmas[2]}", combined)
        key = cv2.waitKey(30)
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()