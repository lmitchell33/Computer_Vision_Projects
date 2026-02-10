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

def temporal_derivative(frames, kernel, std):
    """Applies a 1D differential operator and returns the temporal derivate. #2 in the breakdown of tasks"""
    derivatives = []

    # I am assuming that when the instructions say "temporal derivative" it just means this filter is applies to 
    # time adjacent frames instead of convoluting the pixels in the image (I very likely could be wrong tho)
    for i in range(1, len(frames)-1):
        d_frame = sum([kernel[0]*frames[i-1], kernel[1]*frames[i], kernel[2]*frames[i+1]])
        derivatives.append(d_frame)

    return derivatives

def box_filter(frames,kernel):
    sframes=[]
    for i in range(1, len(frames)-1):
        blur=cv2.filter2D(frames[i],-1,kernel)
        sframes.append(blur)
    return sframes

def gaussian(frames,std):
    sframes=[]
    for i in range(1, len(frames)-1):
        blur=cv2.GaussianBlur(frames[i],(0,0),std)
        sframes.append(blur)
    return  sframes

def main():
    kernel1 = 0.5 * np.array([-1, 0, 1])
    
    std=2.5
    box_filter_size=5
    kernel2 = np.ones((box_filter_size,box_filter_size),np.float32)/(box_filter_size**2)

    frames = read_frames(OFFICE_IMAGES)
    #frames = box_filter(frames,kernel2)
    frames = gaussian(frames, std)
    
    cv2.imshow("Test Image", frames[0])
    cv2.waitKey(0)


if __name__ == "__main__":
    main()