import cv2
import numpy as np

def get_pixel(image, center, x, y):
    new_value = 0
    try:
        if image[x][y]>center:
            new_value = 1
    except:
        pass
    return new_value

def lbp(image):
    height = image.shape[0]
    width = image.shape[1]
    pat = np.zeros((height, width), np.uint8)
    for i in range(height):
        for j in range(width):
            center = image[i, j]
            val_ar = []
            val_ar.append(get_pixel(image, center, i-1, j-1))
            val_ar.append(get_pixel(image, center, i - 1, j))
            val_ar.append(get_pixel(image, center, i - 1, j + 1))
            val_ar.append(get_pixel(image, center, i, j + 1))
            val_ar.append(get_pixel(image, center, i + 1, j + 1))
            val_ar.append(get_pixel(image, center, i + 1, j))
            val_ar.append(get_pixel(image, center, i + 1, j - 1))
            val_ar.append(get_pixel(image, center, i, j - 1))
            val = 0
            for m in range(8):
                val += val_ar[m]*(2**m)
            pat[i,j] = val

    return pat
    

def lbpHistogram(image):
    # Compute the histogram
    histogram, bins = np.histogram(image.flatten(), bins=256, range=[0, 256])

    # Plot the histogram
    plt.figure()
    plt.plot(histogram, color='black')
    plt.xlim([0, 256])
    plt.xlabel('LBP Value')
    plt.ylabel('Frequency')
    plt.title('LBP Histogram')
    plt.show()

if __name__ == '__main__':
    image = cv2.imread("wrinkle_test2.jpg")
    image = cv2.resize(image, (500, 500))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Input", image)
    image = lbp(image)
    cv2.imshow("LBP Output",image)
    lbpHistogram(image)
    cv2.waitKey(0)
