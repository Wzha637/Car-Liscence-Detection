
import math
import sys
from pathlib import Path

from matplotlib import pyplot
from matplotlib.patches import Rectangle
from sqlalchemy import outparam

# import our basic, light-weight png reader library
import imageIO.png

# this function reads an RGB color png file and returns width, height, as well as pixel arrays for r,g,b


def readRGBImageToSeparatePixelArrays(input_filename):

    image_reader = imageIO.png.Reader(filename=input_filename)
    # png reader gives us width and height, as well as RGB data in image_rows (a list of rows of RGB triplets)
    (image_width, image_height, rgb_image_rows,
     rgb_image_info) = image_reader.read()

    print("read image width={}, height={}".format(image_width, image_height))

    # our pixel arrays are lists of lists, where each inner list stores one row of greyscale pixels
    pixel_array_r = []
    pixel_array_g = []
    pixel_array_b = []

    for row in rgb_image_rows:
        pixel_row_r = []
        pixel_row_g = []
        pixel_row_b = []
        r = 0
        g = 0
        b = 0
        for elem in range(len(row)):
            # RGB triplets are stored consecutively in image_rows
            if elem % 3 == 0:
                r = row[elem]
            elif elem % 3 == 1:
                g = row[elem]
            else:
                b = row[elem]
                pixel_row_r.append(r)
                pixel_row_g.append(g)
                pixel_row_b.append(b)

        pixel_array_r.append(pixel_row_r)
        pixel_array_g.append(pixel_row_g)
        pixel_array_b.append(pixel_row_b)

    return (image_width, image_height, pixel_array_r, pixel_array_g, pixel_array_b)


# a useful shortcut method to create a list of lists based array representation for an image, initialized with a value
def createInitializedGreyscalePixelArray(image_width, image_height, initValue=0):

    new_array = [[initValue for x in range(
        image_width)] for y in range(image_height)]
    return new_array


def computeRGBToGreyscale(pixel_array_r, pixel_array_g, pixel_array_b, image_width, image_height):

    greyscale_pixel_array = createInitializedGreyscalePixelArray(
        image_width, image_height)

    # STUDENT CODE HERE
    for x in range(image_height):
        for y in range(image_width):
            r = pixel_array_r[x][y] * 0.299
            g = pixel_array_g[x][y] * 0.587
            b = pixel_array_b[x][y] * 0.115
            g = round(r+g+b)
            greyscale_pixel_array[x][y] = g
    return greyscale_pixel_array


def scaleTo0And255AndQuantize(pixel_array, image_width, image_height):
    minVal = 0
    maxVal = 0
    i = 0
    while i < image_height:
        if i == 0:
            minVal = min(pixel_array[i])
            maxVal = max(pixel_array[i])
        else:
            if min(pixel_array[i]) < minVal:
                minVal = min(pixel_array[i])
            if max(pixel_array[i]) > maxVal:
                maxVal = max(pixel_array[i])
        i += 1

    result = createInitializedGreyscalePixelArray(image_width, image_height)
    interval = maxVal - minVal
    if interval == 0:
        return result
    for x in range(image_height):
        for y in range(image_width):
            output = round(((pixel_array[x][y]-minVal)/interval * 255))
            if output < 0:
                output = 0
            elif output > 255:
                output = 255
            result[x][y] = output
    return result


def computeStandardDeviationImage5x5(pixel_array, image_width, image_height):
    output = createInitializedGreyscalePixelArray(image_width, image_height)

    for y in range(2, image_height-2):
        for x in range(2, image_width-2):
            average = (pixel_array[y-1][x-1] +
                       pixel_array[y-1][x] +
                       pixel_array[y-1][x+1] +
                       pixel_array[y][x-1] +
                       pixel_array[y][x] +
                       pixel_array[y][x+1] +
                       pixel_array[y+1][x-1] +
                       pixel_array[y+1][x] +
                       pixel_array[y+1][x+1] +
                       pixel_array[y-2][x-2] +
                       pixel_array[y-2][x-1] +
                       pixel_array[y-2][x] +
                       pixel_array[y-2][x+1] +
                       pixel_array[y-2][x+2] +
                       pixel_array[y-1][x-2] +
                       pixel_array[y][x-2] +
                       pixel_array[y][x+2] +
                       pixel_array[y+1][x-2] +
                       pixel_array[y+1][x+2] +
                       pixel_array[y+2][x-2] +
                       pixel_array[y+2][x-1] +
                       pixel_array[y+2][x] +
                       pixel_array[y+2][x+1] +
                       pixel_array[y+1][x+2] +
                       pixel_array[y][x])

            average /= 25
            variance = (pow((pixel_array[y-1][x-1] - average), 2) +
                        pow((pixel_array[y-1][x] - average), 2) +
                        pow((pixel_array[y-1][x+1] - average), 2) +
                        pow((pixel_array[y][x-1] - average), 2) +
                        pow((pixel_array[y][x] - average), 2) +
                        pow((pixel_array[y][x+1] - average), 2) +
                        pow((pixel_array[y+1][x-1] - average), 2) +
                        pow((pixel_array[y+1][x] - average), 2) +
                        pow((pixel_array[y+1][x+1] - average), 2) +
                        pow((pixel_array[y-2][x-2] - average), 2) +
                        pow((pixel_array[y-2][x-1] - average), 2) +
                        pow((pixel_array[y-2][x] - average), 2) +
                        pow((pixel_array[y-2][x+1] - average), 2) +
                        pow((pixel_array[y-2][x+2] - average), 2) +
                        pow((pixel_array[y-1][x-2] - average), 2) +
                        pow((pixel_array[y][x-2] - average), 2) +
                        pow((pixel_array[y][x+2] - average), 2) +
                        pow((pixel_array[y+1][x-2] - average), 2) +
                        pow((pixel_array[y+1][x+2] - average), 2) +
                        pow((pixel_array[y+2][x-2] - average), 2) +
                        pow((pixel_array[y+2][x-1] - average), 2) +
                        pow((pixel_array[y+2][x] - average), 2) +
                        pow((pixel_array[y+2][x+1] - average), 2) +
                        pow((pixel_array[y+1][x+2] - average), 2) +
                        pow((pixel_array[y][x] - average), 2))

            variance /= 25
            sd = math.sqrt(variance)
            output[y][x] = round(sd.real, 2)
    return output


def computeThresholdGE(pixel_array, threshold_value, image_width, image_height):
    histogram = []
    for x in range(image_height):
        histogram.append([])
        for y in range(image_width):
            if(pixel_array[x][y] >= threshold_value):
                histogram[x].append(255)
            else:
                histogram[x].append(0)
    return histogram


def computeDilation8Nbh3x3FlatSE(pixel_array, image_width, image_height):
    result = createInitializedGreyscalePixelArray(image_width, image_height)
    output = createInitializedGreyscalePixelArray(image_width, image_height)
    for x in range(1, image_height-1):
        for y in range(1, image_width-1):
            output[x][y] = pixel_array[x][y]
    for i in range(1, image_height-1):
        for j in range(1, image_width-1):
            list = []
            for x in [-1, 0, 1]:
                for y in [-1, 0, 1]:
                    list.append(output[x+i][y+j])
            if 1 in list or 255 in list:
                result[i-1][j-1] = 1
            else:
                result[i-1][j-1] = 0
    return result


def computeErosion8Nbh3x3FlatSE(pixel_array, image_width, image_height):
    result = createInitializedGreyscalePixelArray(image_width, image_height)

    for x in range(1, image_height-1):
        for y in range(1, image_width-1):
            sum = 0

            square = [[pixel_array[x-1][y+1], pixel_array[x][y+1], pixel_array[x+1][y+1]],
                      [pixel_array[x-1][y], pixel_array[x][y], pixel_array[x+1][y]],
                      [pixel_array[x-1][y-1], pixel_array[x]
                       [y-1], pixel_array[x+1][y-1]]
                      ]

            for i in range(3):
                for j in range(3):
                    sum += square[i][j]

            if sum / 9 == square[0][0] and sum / 9 != 0:
                result[x][y] = 1

    return result


class Queue:
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def enqueue(self, item):
        self.items.insert(0, item)

    def dequeue(self):
        return self.items.pop()

    def size(self):
        return len(self.items)


def computeConnectedComponentLabeling(pixel_array, image_width, image_height):
    visited = [[0 for x in range(image_width)] for y in range(image_height)]
    label = 0
    labelDict = {}
    for x in range(image_height):
        for y in range(image_width):
            if pixel_array[x][y] != 0 and visited[x][y] == 0:
                queue = Queue()
                label += 1
                visited[x][y] = 1
                labelDict[label] = 1
                queue.enqueue((x, y))
                while queue.size() != 0:
                    index = queue.dequeue()
                    pixel_array[index[0]][index[1]] = label
                    if pixel_array[index[0] - 1][index[1]] != 0 and visited[index[0] - 1][index[1]] == 0:
                        queue.enqueue((index[0] - 1, index[1]))
                        visited[index[0] - 1][index[1]] = 1
                        labelDict[label] += 1

                    if pixel_array[index[0]][index[1] - 1] != 0 and visited[index[0]][index[1] - 1] == 0:
                        queue.enqueue((index[0], index[1] - 1))
                        visited[index[0]][index[1] - 1] = 1
                        labelDict[label] += 1

                    if pixel_array[index[0]][index[1] + 1] != 0 and visited[index[0]][index[1] + 1] == 0:
                        queue.enqueue((index[0], index[1] + 1))
                        visited[index[0]][index[1] + 1] = 1
                        labelDict[label] += 1

                    if pixel_array[index[0] + 1][index[1]] != 0 and visited[index[0] + 1][index[1]] == 0:
                        queue.enqueue((index[0] + 1, index[1]))
                        visited[index[0] + 1][index[1]] = 1
                        labelDict[label] += 1

    return pixel_array, labelDict


def setBoundary(pixel_array, image_width, image_height, labelDict):
    max_connect = 1
    for i in labelDict.keys():
        if labelDict.get(i) > labelDict.get(max_connect):
            max_connect = i
    minX = image_width
    minY = image_height
    maxX = 0
    maxY = 0

    for y in range(image_height):
        for x in range(image_width):
            if (pixel_array[y][x] == max_connect):
                if minX > x:
                    minX = x
                if minY > y:
                    minY = y
                if maxX < x:
                    maxX = x
                if maxY < y:
                    maxY = y

    return minX, maxX, minY, maxY
# This is our code skeleton that performs the license plate detection.
# Feel free to try it on your own images of cars, but keep in mind that with our algorithm developed in this lecture,
# we won't detect arbitrary or difficult to detect license plates!


def main():

    command_line_arguments = sys.argv[1:]

    SHOW_DEBUG_FIGURES = True

    # this is the default input image filename
    input_filename = "numberplate5.png"

    if command_line_arguments != []:
        input_filename = command_line_arguments[0]
        SHOW_DEBUG_FIGURES = False

    output_path = Path("output_images")
    if not output_path.exists():
        # create output directory
        output_path.mkdir(parents=True, exist_ok=True)

    output_filename = output_path / \
        Path(input_filename.replace(".png", "_output.png"))
    if len(command_line_arguments) == 2:
        output_filename = Path(command_line_arguments[1])

    # we read in the png file, and receive three pixel arrays for red, green and blue components, respectively
    # each pixel array contains 8 bit integer values between 0 and 255 encoding the color values
    (image_width, image_height, px_array_r, px_array_g,
     px_array_b) = readRGBImageToSeparatePixelArrays(input_filename)

    # setup the plots for intermediate results in a figure
    fig1, axs1 = pyplot.subplots(2, 2)
    axs1[0, 0].set_title('Input red channel of image')
    axs1[0, 0].imshow(px_array_r, cmap='gray')
    axs1[0, 1].set_title('Input green channel of image')
    axs1[0, 1].imshow(px_array_g, cmap='gray')
    axs1[1, 0].set_title('Input blue channel of image')
    axs1[1, 0].imshow(px_array_b, cmap='gray')

    # STUDENT IMPLEMENTATION here

    g_array = computeRGBToGreyscale(px_array_r, px_array_g, px_array_b, image_width, image_height)
    px_array = scaleTo0And255AndQuantize(g_array, image_width, image_height)
    sd_array = computeStandardDeviationImage5x5(px_array, image_width, image_height)
    px_array = scaleTo0And255AndQuantize(sd_array, image_width, image_height)
    px_array = computeThresholdGE(px_array, 150, image_width, image_height)
    px_array = computeDilation8Nbh3x3FlatSE(px_array, image_width, image_height)
    px_array = computeDilation8Nbh3x3FlatSE(px_array, image_width, image_height)
    px_array = computeDilation8Nbh3x3FlatSE(px_array, image_width, image_height)
    px_array = computeDilation8Nbh3x3FlatSE(px_array, image_width, image_height)
    px_array = computeErosion8Nbh3x3FlatSE(px_array, image_width, image_height)
    px_array = computeErosion8Nbh3x3FlatSE(px_array, image_width, image_height)
    px_array = computeErosion8Nbh3x3FlatSE(px_array, image_width, image_height)
    px_array = computeErosion8Nbh3x3FlatSE(px_array, image_width, image_height)
    (pixel_array, labelDict) = computeConnectedComponentLabeling(px_array, image_width, image_height)
    (minX, maxX, minY, maxY) = setBoundary(pixel_array, image_width, image_height, labelDict)
    g_array = computeRGBToGreyscale(px_array_r, px_array_g, px_array_b, image_width, image_height)
    px_array = scaleTo0And255AndQuantize(g_array, image_width, image_height)
    # compute a dummy bounding box centered in the middle of the input image, and with as size of half of width and height
    # center_x = image_width / 2.0
    # center_y = image_height / 2.0
    # bbox_min_x = center_x - minX / 4.0
    # bbox_max_x = center_x + maxX / 4.0
    # bbox_min_y = center_y - minY / 4.0
    # bbox_max_y = center_y + maxY / 4.0

    # Draw a bounding box as a rectangle into the input image
    axs1[1, 1].set_title('Final image of detection')
    axs1[1, 1].imshow(px_array, cmap='gray')
    rect = Rectangle((minX, minY), maxX - minX, maxY - minY, linewidth=1,
                     edgecolor='g', facecolor='none')
    axs1[1, 1].add_patch(rect)

    # write the output image into output_filename, using the matplotlib savefig method
    extent = axs1[1, 1].get_window_extent().transformed(
        fig1.dpi_scale_trans.inverted())
    pyplot.savefig(output_filename, bbox_inches=extent, dpi=600)

    if SHOW_DEBUG_FIGURES:
        # plot the current figure
        pyplot.show()


if __name__ == "__main__":
    main()
