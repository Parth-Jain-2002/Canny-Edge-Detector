# import libraries
from skimage import data,io,filters,feature
from matplotlib import pyplot as plt 
from skimage.color import rgb2gray 
from skimage.feature import canny
from skimage.metrics import structural_similarity as ssim
from math import log10,sqrt
import scipy.ndimage as ndi
import numpy as np
import sys

# convolution of two matrixes of same shape
def convolution(matrix1,matrix2):
    sum = 0
    height, width = matrix1.shape
    for row in range(0,height):
        for col in range(0,width):
            sum = sum + matrix1[row,col]*matrix2[row,col]
    return sum

# gaussian filter having size of (2*half_size + 1) and sigma as parameter
def gaussian_filter(half_size,sigma = 1):
    size = 2 * half_size + 1
    gaussian_kernel = np.zeros(shape=(size,size))
    coeff = 1/(2.0 * np.pi * (sigma**2))
    for i in range(0,size):
        for j in range(0,size):
            x = i - half_size
            y = j - half_size
            exponent_term = np.exp(-((x**2 + y**2)/(2.0 * (sigma**2))))
            gaussian_kernel[i,j] = exponent_term * coeff
    return gaussian_kernel/np.sum(gaussian_kernel)
    

def gaussian_blur(image,half_size,sigma = 1):
    gaussian_kernel = gaussian_filter(half_size,sigma)
    height, width = image.shape
    
    resultant = np.zeros(shape = (height,width))
    for i in range(half_size,height-half_size):
        for j in range(half_size,width-half_size):
            resultant[i,j] = convolution(gaussian_kernel, image[i-half_size:i+half_size+1,j-half_size:j+half_size+1])
    
    # Padding in all boundary directions

    # Top
    for i in range(half_size-1,-1,-1):
        for j in range(1,width-1):
            resultant[i,j]=resultant[i+1,j]
    
    # Bottom
    for i in range(height-half_size,height):
        for j in range(1,width-1):
            resultant[i,j]=resultant[i-1,j]
    
    # Left
    for j in range(half_size-1,0,-1):
        for i in range(0,height):
            resultant[i,j]=resultant[i,j+1]
    
    # Right
    for j in range(width-half_size,width):
        for i in range(0,height):
            resultant[i,j]=resultant[i,j-1]

    return resultant

def sobel_filter(image):
    # If image is not grayscale, converting it to grayscale
    if(len(image.shape)==3):
      image = rgb2gray(image)

    height, width = image.shape
    output = np.zeros(shape = (height,width))
    angle = np.zeros(shape = (height,width))
    gradientX = np.zeros(shape = (height,width))
    gradientY = np.zeros(shape = (height,width))
    sobelYFilter = np.array([[1,2,1],
                            [0,0,0],
                            [-1,-2,-1]])
    sobelXFilter = np.array([[-1,0,1],
                            [-2,0,2],
                            [-1,0,1]])
    for i in range(1,height-1):
      for j in range(1,width-1):
           # convolution with Sobel X Filter
           gradientX[i,j] = convolution(sobelXFilter,image[i-1:i+2,j-1:j+2])
           # convolution with Sobel Y Filter
           gradientY[i,j] = convolution(sobelYFilter,image[i-1:i+2,j-1:j+2])

           # Contribution from the results of both Sobel X and Sobel Y
           output[i,j] = np.sqrt(gradientX[i,j]**2 + gradientY[i,j]**2)

           # Calculating the gradient, the principal orientation
           angle[i,j] = np.degrees(np.arctan2(gradientY[i,j],gradientX[i,j]))

    output = output * 255.0/ output.max()
    return output, angle



def non_max_suppression(image,dir):
    # Reference : https://www.southampton.ac.uk/~msn/book/new_demo/nonmax/ 
    # From the reference : The image is scanned along the image gradient direction, 
    # and if pixels are not part of the local maxima they are set to zero. This has 
    # the effect of supressing all image information that is not part of local maxima.

    height, width = image.shape
    resultant = np.zeros(image.shape)
 
    for row in range(1, height - 1):
        for col in range(1, width - 1):
            direction = dir[row, col]

            if(direction<0):
                direction= direction + 180
            
        #   | 1 | 2 | 3 |
        #   | 4 | 5 | 6 |
        #   | 7 | 8 | 9 |

            if (0 <= direction < 180 / 8) or (7 * 180 / 8 <= direction <= 180):
                # Checking if 5 is local maxima for the patch [4,5,6] since gradient is in that direction
                if image[row, col] >= image[row, col - 1] and image[row, col] >= image[row, col + 1]:
                    resultant[row, col] = image[row, col]

            elif (180 / 8 <= direction < 3 * 180 / 8):
                # Checking if 5 is local maxima for the patch [3,5,7] since gradient is in that direction
                if image[row, col] >= image[row+1, col - 1] and image[row, col] >= image[row-1, col + 1]:
                    resultant[row, col] = image[row, col]
 
            elif (3 * 180 / 8 <= direction < 5 * 180 / 8):
                # Checking if 5 is local maxima for the patch [2,5,8] since gradient is in that direction
                if image[row, col] >= image[row-1, col] and image[row, col] >= image[row+1, col]:
                    resultant[row, col] = image[row, col]
 
            else:
                # Checking if 5 is local maxima for the patch [1,5,9] since gradient is in that direction
                if image[row, col] >= image[row-1, col-1] and image[row, col] >= image[row+1, col+1]:
                    resultant[row, col] = image[row, col]
 
    return resultant
 
 
def threshold(image, low, high):
    resultant = np.zeros(image.shape)
    height, width = image.shape

    for i in range(0,height):
        for j in range(0,width):
            # marking as strong edge
            if image[i,j]>=high:
                resultant[i,j]=255  # strong
            # marking as weak edge
            elif image[i,j]<=high and image[i,j]>=low:
                resultant[i,j]=50   # weak

    return resultant
 
 
def hysteresis(image):
    # If weak edge is connected to strong edge, we make it strong
    # We go in 4 directions, right to left, left to right, top to bottom, bottom to right
    # we cover all paths essentially that could have made

    # In hystersis, we declare a weak edge as strong if it is connected to a strong edge

    height, width = image.shape
    tb = image.copy()
    bt = image.copy()
    rl = image.copy()
    lr = image.copy()

    dx = np.array([0,0,1,-1,1,-1,1,-1])
    dy = np.array([1,-1,0,0,1,-1,-1,1])
    
    # right to left
    for row in range(1,height):
        for col in range(width-1,0,-1):
            if rl[row, col] == 50:
                flag = False
                for i in range(0,8):
                    t_row = row + dx[i]
                    t_col = col + dy[i]
                    if(rl[t_row,t_col]==255):
                        rl[row,col]=255
                        flag = True
                if flag == False:
                    rl[row,col]=0
    
    # left to right
    for row in range(height - 1, 0, -1):
        for col in range(1, width):
            if lr[row, col] == 50:
                flag = False
                for i in range(0,8):
                    t_row = row + dx[i]
                    t_col = col + dy[i]
                    if(lr[t_row,t_col]==255):
                        lr[row,col]=255
                        flag = True
                if flag == False:
                    lr[row,col]=0
 
    # top to bottom
    for row in range(1, height):
        for col in range(1, width):
            if tb[row, col] == 50:
                flag = False
                for i in range(0,8):
                    t_row = row + dx[i]
                    t_col = col + dy[i]
                    if(tb[t_row,t_col]==255):
                        tb[row,col]=255
                        flag = True
                if flag == False:
                    tb[row,col]=0
    
    # bottom to top
    for row in range(height-1,0,-1):
        for col in range(width-1,0,-1):
            if bt[row, col] == 50:
                flag = False
                for i in range(0,8):
                    t_row = row + dx[i]
                    t_col = col + dy[i]
                    if(bt[t_row,t_col]==255):
                        bt[row,col]=255
                        flag = True
                if flag == False:
                    bt[row,col]=0
    
    # Taking all the different components
    resultant = tb + bt + rl + lr 

    count = 0
    for i in range(0,height):
        for j in range(0,width):
            if resultant[i,j] > 255:
                resultant[i,j] = 255
    return resultant

# To compute peak signal to noise ratio between two images
def PSNR(image1,image2):
    # Reference : https://in.mathworks.com/help/vision/ref/psnr.html 
    height,width = image1.shape
    mse = 0
    for row in range(0,height):
        for col in range(0,width):
            mse = mse + (image1[row,col]-image2[row,col])**2

    mse = mse / (height * width)
    if(mse == 0):
        return 100
    max_px = 255.0
    psnr = 20 * log10(max_px/sqrt(mse))
    return psnr


def myCannyEdgeDetector(image,low_threshold,high_threshold):

    if(len(image.shape)==3):
      image = rgb2gray(image)
    
    # Step 1 : Applying the gaussian filter
    smooth_img = gaussian_blur(image,2,1.5)
    
    # Step 2 : Sobel Detection
    sobel_img, gradient_dir = sobel_filter(smooth_img)
    
    # Step 3 : Non - Maximal Suppression
    non_max_out = non_max_suppression(sobel_img,gradient_dir)
    
    # Step 4 : Double Thresholding
    threshhold_out = threshold(non_max_out,low_threshold,high_threshold)

    # Step 5 : Linking via hystersis
    hysteresis_out = hysteresis(threshhold_out)

    return hysteresis_out

if __name__ == "__main__":

    argument = len(sys.argv)
    if argument == 1:
        # Default image
        inputImage = data.astronaut()
    else:
        # To take input from the command line
        inputImage = io.imread(sys.argv[1])
    
    # If the image has 3 dimensions, converting it to grayscale
    if(len(inputImage.shape)==3):
      image = rgb2gray(inputImage)
    else:
      image = inputImage

    # Edge Map by my canny edge detector
    # Discussed about the approach behind deciding this thresholds
    edgeMap_using_my_canny = myCannyEdgeDetector(image,5,20)

    # Edge Map by the inbuilt canny edge detector
    edgeMap_using_inbuilt_canny = canny(image,sigma = 1.5)
             
    # PSNR Score
    PSNRScore = PSNR(edgeMap_using_inbuilt_canny,edgeMap_using_my_canny)
    print("The PSNR (Peak Signal to Noise Ratio) between the outputs of the images : ",PSNRScore)
    
    # SSIM Score
    (SSIMScore, diff) = ssim(edgeMap_using_my_canny.astype('bool'),edgeMap_using_inbuilt_canny.astype('bool'),full = True)
    print("The SSIM (Structural Similarity Index Metric) between the outputs of the images : ",SSIMScore)
    

    # Displaying the original image, output of inbuilt canny, output of myCanny detector, SSIM and PSNR
    fig, axes = plt.subplots(1,3,figsize=(16,8))

    axes[0].imshow(edgeMap_using_inbuilt_canny,cmap="gray")
    axes[1].imshow(edgeMap_using_my_canny,cmap = "gray")
    axes[2].imshow(inputImage,cmap = "gray")
    axes[0].set_title("Edge Map using inbuilt canny edge detector")
    axes[1].set_title("Edge Map using my canny edge detector")
    axes[2].set_title("Original Image")
    axes[0].axis('off')
    axes[1].axis('off')
    axes[2].axis('off')
    text = "The SSIM (Structural Similarity Index Metric) between the outputs of the images : " + str(SSIMScore)
    text = text + "\nThe PSNR (Peak Signal to Noise Ratio) between the outputs of the images : " + str(PSNRScore)
    plt.figtext(0.05,0.05,text)
    plt.show()