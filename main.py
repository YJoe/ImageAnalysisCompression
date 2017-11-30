import numpy as np
import cv2
import time
import re

class Node:
    def __init__(self, left_node, right_node, value, frequency):
        self.left_node = left_node
        self.right_node = right_node
        self.value = value
        self.frequency = frequency

    def print_details(self):
        if self.left_node is not None:
            print "left      [", self.left_node.value, "]"
        else:
            print "left      [ IS NULL ]"
        if self.right_node is not None:
            print "right     [", self.right_node.value, "]"
        else:
            print "right     [ IS NULL ]"

        print "value     [", self.value, "]"
        print "frequency [", self.frequency, "]"


def print_network(root, tab):
    print tab, "[", root.value, ":", root.frequency, "]"
    if root.left_node is not None:
        print_network(root.left_node, tab + "\t0")
    if root.right_node is not None:
        print_network(root.right_node, tab + "\t1")


def get_frequency_list(data):
    frequency_data = []

    # for all of the data elements we have
    for i in range(0, len(data)):

        # assume we haven't found the element in the frequency list yet
        found_index = -1

        # look through all elements in the frequency list
        for j in range(0, len(frequency_data)):
            if data[i] == frequency_data[j][0]:
                # store the index we found the element
                found_index = j
                break

        # if we found the letter in the array add to the current count
        if found_index > -1:
            frequency_data[found_index][1] += 1

        # else we need to add a new element to the frequency list
        else:
            frequency_data.append([data[i], 1])

    # sort the frequency data by the frequency so that lowest is first
    frequency_data = sorted(frequency_data, key=lambda element: element[1])

    return frequency_data


def create_nodes(frequency_data):
    node_data = []

    for f in frequency_data:
        node_data.append(Node(None, None, f[0], f[1]))

    return node_data


def append_to_node_tree(root, node_values):
    #print "\t\t\tTrying to append to root [", root.value, "(", root.frequency, ")] something form [", ",".join(str(n.value) + "(" + str(n.frequency) + ")" for n in node_values), "]"
    for i in range(0, len(node_values)):
        if node_values[0].frequency >= root.frequency:
            #print "\t\t\t\tThe next node frequency [", node_values[0].frequency, "] is >= [", root.frequency, "]"
            left_node = node_values[0]
            del node_values[0]
            root = Node(left_node, root, str(left_node.value + root.value), left_node.frequency + root.frequency)
            #print "\t\t\t\t\tAppended left node [", left_node.value, "] to existing tree"
            return append_to_node_tree(root, node_values)

    # return the new root node and the node values that remain, if there are no node values then return an empty []
    return root, [] if node_values is None else node_values


def create_tree(node_values):

    #print "\nCreating tree"

    roots = []

    # while there are still frequency values that haven't been appended to a tree
    while len(node_values) != 0:

        #print "\tThere are still frequency values [", ",".join(str(n.value) + "(" + str(n.frequency) + ")" for n in node_values), "]"

        # if we can't make a pair then this single element should be in its own tree
        if len(node_values) == 1:
            #print "\t\tWe can't make a pair though so placing [", node_values[0].value, "] in its own tree"
            root = node_values[0]
            del node_values[0]

        # we can make a pair so we will make a tree with them
        else:
            #print "\t\tThere are at least two nodes left, creating pair"
            # start a tree
            left = node_values[1]
            right = node_values[0]
            root = Node(left, right, str(left.value + "" + right.value), left.frequency + right.frequency)

            # remove the two nodes that are now in the tree
            del node_values[0]
            del node_values[0]

            root, node_values = append_to_node_tree(root, node_values)

        roots.append(root)

    if len(roots) != 1:
        # make sure that our roots are still ordered by frequency
        roots = sorted(roots, key=lambda n: n.frequency)
        roots = create_tree(roots)

    return roots


def encode_char(root, char, code):

    if char == root.value:
        return code

    elif char in root.left_node.value:
        code += "0"
        return encode_char(root.left_node, char, code)

    elif char in root.right_node.value:
        code += "1"
        return encode_char(root.right_node, char, code)

    else:
        print "ERROR WHILE DECODING"
        return "?"


def encode(root, regular_text):
    return "".join(encode_char(root, letter, "") for letter in regular_text)


def decode_char(root, code):
    if len(root.value) == 1:
        return root.value
    elif code == "":
        return None
    elif code[0] == "1":
        return decode_char(root.right_node, code[1:])
    else:
        return decode_char(root.left_node, code[1:])


def decode(root, code):
    output = ""

    while len(code) != 0:
        for i in range(1, 10):
            d = decode_char(root, code[0:i])
            if d is not None:
                output += d
                code = code[i:]
                break

    return output


def check_for_code(full_code, code_size, direc):
    current_code = full_code[0:code_size]
    if current_code in direc:
        return full_code[i:], direc[current_code]

    return None


def directory_decode(root, direct, code):
    output = ""

    min = 100
    max = 0

    for dire in direct:
        if len(dire) < min:
            min = len(dire)
        if len(dire) > max:
            max = len(dire)

    while len(code) != 0:
        for i in range(min, max + 1):
            current_code = code[0:i]
            if current_code in direct:
                code = code[i:]
                output += direct[current_code]
                break

    return output


def get_decode_directory(root):
    directory = []

    for char in root.value:
        directory.append([encode(root, char), char])

    return dict(directory)


def image_to_string(image):

    height, width, depth = image.shape
    flat = image.flatten()

    image_string = "[" + str(width) + "," + str(height) + "," + str(depth) + "]"

    for i in range(0, len(flat), 3):
        image_string += "[" + ",".join(str(flat[i + j]) for j in range(0, 3)) + "]"

    return image_string


def string_to_image(string):

    arr = re.findall("\[(.*?)\]", string)

    # the first set is [width, height, depth] of the image
    width, height, depth = arr[0].split(",")
    width = int(width)
    height = int(height)
    depth = int(depth)
    print "Width", width, " Height", height, " Depth", depth

    # create a blank image with the dimensions specified by the header
    blank_image = np.zeros((height, width, depth), np.uint8)

    # get rid of the first bit of metadata because it isn't actually a pixel
    arr = arr[1:]

    # populate the image with the pixel values defined by the string
    for i in range(0, width):
        for j in range(0, height):
            blank_image[i][j] = [int(arr[j + i * width].split(',')[0]), int(arr[j + i * width].split(',')[1]), int(arr[j + i * width].split(',')[2])]

    return blank_image


def split_blocks(image, block_width, block_height):
    image_segs = []
    for i in range(0, image.shape[0], block_height):
        for j in range(0, image.shape[1], block_width):
            image_segs.append(image[i:i+block_height, j:j+block_width])

    return image_segs


def combine_blocks(image_segs, image_width, image_height, sample_width, sample_height):
    x = image_width / sample_width
    y = image_height / sample_height

    # create a blank image of the size of the combined image
    combined_image = np.zeros((y * sample_width, x * sample_height, 3), np.uint8)

    k = 0
    for i in range(0, y):
        for j in range(0, x):
            combined_image[i * sample_width: i * sample_width + sample_width, j * sample_height: j * sample_height + sample_height] = image_segs[k]
            k += 1

    return combined_image


def compress_block(block, quantization_table):

    # convert to from uint to int so that we can shift to negative values
    block = np.array(block, int)

    # subtract from every channel of every pixel
    block = block - 128

    # scale every channel of every pixel down within range -1 -> 1
    block = block / 255.0

    # split the block into three channels
    block_channels = cv2.split(block)

    # take dct of each block channel
    for c in range(0, 3):
        block_channels[c] = cv2.dct(block_channels[c])

    # quantization
    for c in range(0, 3):
        block_channels[c] = block_channels[c] / quantization_table[c]
        block_channels[c] = block_channels[c] * 255
        block_channels[c] = block_channels[c].astype(int)
        block_channels[c] = block_channels[c] / 255.0

    # inverse quantisation
    for c in range(0, 3):
        block_channels[c] = block_channels[c] * quantization_table[c]

    # take inverse dtc of block channel
    for c in range(0, 3):
        block_channels[c] = cv2.idct(block_channels[c])

    # merge the block channels back into one ycrbr image
    block = cv2.merge(block_channels)

    # scale up by 255
    block = block * 255

    # shift range to be 0 - 255
    block = block + 128

    for i in range(0, block.shape[0]):
        for j in range(0, block.shape[1]):
            for c in range(0, 3):
                if block[i][j][c] > 255:
                    block[i][j][c] = 255
                elif block[i][j][c] < 0:
                    block[i][j][c] = 0

    block = block.astype(np.uint8)

    return block


if __name__ == "__main__":

    #img = cv2.imread('images/file.png', 1)
    img = cv2.imread('images/turtle.jpg', 1)
    #img = cv2.imread('images/fish.jpg', 1)

    cv2.imshow("ORIGINAL", img)

    dct_width = 8
    dct_height = 8

    border_w = img.shape[0] % dct_width
    border_h = img.shape[1] % dct_height

    print "ADDING BORDER WH[", border_w, ", ", border_h, "]"

    img = cv2.copyMakeBorder(img, 0, border_h, 0, border_w, cv2.BORDER_CONSTANT)

    image_width = img.shape[0]
    image_height = img.shape[1]

    # Taken from [https://www.hdm-stuttgart.de/~maucher/Python/MMCodecs/html/jpegUpToQuant.html]
    QY = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                   [12, 12, 14, 19, 26, 48, 60, 55],
                   [14, 13, 16, 24, 40, 57, 69, 56],
                   [14, 17, 22, 29, 51, 87, 80, 62],
                   [18, 22, 37, 56, 68, 109, 103, 77],
                   [24, 35, 55, 64, 81, 104, 113, 92],
                   [49, 64, 78, 87, 103, 121, 120, 101],
                   [72, 92, 95, 98, 112, 100, 103, 99]], dtype=int)

    # Taken from [https://www.hdm-stuttgart.de/~maucher/Python/MMCodecs/html/jpegUpToQuant.html]
    QC = np.array([[17, 18, 24, 47, 99, 99, 99, 99],
                   [18, 21, 26, 66, 99, 99, 99, 99],
                   [24, 26, 56, 99, 99, 99, 99, 99],
                   [47, 66, 99, 99, 99, 99, 99, 99],
                   [99, 99, 99, 99, 99, 99, 99, 99],
                   [99, 99, 99, 99, 99, 99, 99, 99],
                   [99, 99, 99, 99, 99, 99, 99, 99],
                   [99, 99, 99, 99, 99, 99, 99, 99]], dtype=int)

    # Taken from [https://www.hdm-stuttgart.de/~maucher/Python/MMCodecs/html/jpegUpToQuant.html]
    QF = 10.0
    if QF < 50 and QF > 1:
        scale = np.floor(5000 / QF)
    elif QF < 100:
        scale = 200 - 2 * QF
    else:
        print "Quality Factor must be in the range [1..99]"
    scale = scale / 100.0

    # store each channel quantisation table
    Q = [QY * scale, QC * scale, QC * scale]

    # convert from BGR to YCrBr colour space
    image_ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

    # divide the image into several 8*8*3 images
    image_blocks = split_blocks(image_ycc, dct_width, dct_height)

    # compress blocks using the quantisation tables made eariler
    compressed_blocks = []
    for i in range(0, len(image_blocks)):
        compressed_blocks.append(compress_block(image_blocks[i], Q))

    # stitch together all of the compressed blocks
    image_ycc = combine_blocks(compressed_blocks, image_width, image_height, dct_width, dct_height)

    # convert the new compressed image back into the BGR colour space
    image = cv2.cvtColor(image_ycc, cv2.COLOR_YCrCb2BGR)

    # trim off the border we crated earlier
    image = image[0: image_width - border_w , 0: image_height - border_h]

    cv2.imshow("COMPRESSED", image)
    cv2.waitKey()


if __name__ == "__main2__":

    img = cv2.imread('images/turtle.jpg', 1)

    cv2.imshow("Original", img)
    cv2.waitKey()

    start_time = time.time()
    text = image_to_string(img)
    frequency_list = get_frequency_list(text)
    node_list = create_nodes(frequency_list)
    root_node = create_tree(node_list)[0]
    end_time = time.time()
    print "Tree creation time [", end_time - start_time, "]"

    print "\nEncoding string"
    start_time = time.time()
    encoded_text = encode(root_node, text)
    end_time = time.time()
    print "Encode time [", end_time - start_time, "]"

    print "\nDecoding string"
    start_time = time.time()
    #decoded_text = decode(root_node, encoded_text)
    decode_dir = get_decode_directory(root_node)
    print "DONE"
    decoded_text = directory_decode(root_node, decode_dir, encoded_text)
    end_time = time.time()
    print "Decode time [", end_time - start_time, "]"

    print "\nString to image"
    start_time = time.time()
    decoded_image = string_to_image(decoded_text)
    end_time = time.time()
    print "String to image time [", end_time - start_time, "]"

    cv2.imshow("Decoded", decoded_image)
    cv2.waitKey()
    cv2.destroyAllWindows()

    print "\nDone"



