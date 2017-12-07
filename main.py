import numpy as np
import cv2
import time
import re
import struct
import os


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
        node_data.append(Node(None, None, [f[0]], f[1]))

    return node_data


def get_block_channel_values(block):

    channel_values = []

    # get the size of the block (should be 8)
    n = len(block)
    j = 0
    k = 0

    # for all of the left and bottom edge indexes for this n
    for i in range(0, n * 2 - 1):

        # for the current edge loop through the diagonal from this point
        x = j
        y = k
        while x >= 0 and y < len(block[0]):
            for l in range(0, 3):
                channel_values.append(block[x][y][l])
            x -= 1
            y += 1

        # if i is still less than the block height keep iterating up
        if i < n - 1:
            j += 1

        # i reached the mid point, start moving the other way
        else:
            k += 1

    return channel_values


def create_tree(node_values):

    # as long as we have two more nodes
    while len(node_values) > 1:

        # pick the two smallest nodes
        left = node_values[0]
        right = node_values[1]

        # create a new node from the two
        root = Node(left, right, left.value + right.value, left.frequency + right.frequency)

        # delete the two nodes we just used up
        del node_values[0]
        del node_values[0]

        node_values.append(root)
        node_values = sorted(node_values, key=lambda n: n.frequency)

    print_network(node_values[0], "")

    return node_values[0]


def encode_number(root, number, code):
    if len(root.value) == 1:
        return code

    elif number in root.left_node.value:
        code += "0"
        return encode_number(root.left_node, number, code)

    elif number in root.right_node.value:
        code += "1"
        return encode_number(root.right_node, number, code)

    else:
        print "ERROR WHILE DECODING"
        return "?"


def encode(root, values):
    encoded_string = ""

    return "".join(encode_number(root, v, "") for v in values)


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
    output = []

    min = 100
    max = 0

    for dire in direct:
        if len(dire) < min:
            min = len(dire)
        if len(dire) > max:
            max = len(dire)

    code_index = 0
    while code_index < len(code):
        for i in range(min, max + 1):
            current_code = code[code_index:code_index + i]
            if current_code in direct:
                code_index += i
                output.append(direct[current_code])
                break

    return output


def get_decode_directory(root):
    directory = []

    for v in root.value:
        directory.append([encode_number(root, v, ""), v])

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


def get_quantisation_channels(quality_factor):

    # array was found here [https://www.hdm-stuttgart.de/~maucher/Python/MMCodecs/html/jpegUpToQuant.html]
    QY = np.array([[16, 11, 10, 16, 24,  40,  51,  61 ],
                   [12, 12, 14, 19, 26,  48,  60,  55 ],
                   [14, 13, 16, 24, 40,  57,  69,  56 ],
                   [14, 17, 22, 29, 51,  87,  80,  62 ],
                   [18, 22, 37, 56, 68,  109, 103, 77 ],
                   [24, 35, 55, 64, 81,  104, 113, 92 ],
                   [49, 64, 78, 87, 103, 121, 120, 101],
                   [72, 92, 95, 98, 112, 100, 103, 99 ]], dtype=int)

    # array was found here [https://www.hdm-stuttgart.de/~maucher/Python/MMCodecs/html/jpegUpToQuant.html]
    QC = np.array([[17, 18, 24, 47, 99, 99, 99, 99],
                   [18, 21, 26, 66, 99, 99, 99, 99],
                   [24, 26, 56, 99, 99, 99, 99, 99],
                   [47, 66, 99, 99, 99, 99, 99, 99],
                   [99, 99, 99, 99, 99, 99, 99, 99],
                   [99, 99, 99, 99, 99, 99, 99, 99],
                   [99, 99, 99, 99, 99, 99, 99, 99],
                   [99, 99, 99, 99, 99, 99, 99, 99]], dtype=int)

    # scaling found here [https://www.hdm-stuttgart.de/~maucher/Python/MMCodecs/html/jpegUpToQuant.html]
    if quality_factor < 50 and quality_factor > 1:
        scale = np.floor(5000 / quality_factor)
    elif quality_factor < 100:
        scale = 200 - 2 * quality_factor
    else:
        print "Quality Factor must be in the range [1..99]"
        exit()

    scale = scale / 100.0

    # store each channel quantisation table
    return [QY * scale, QC * scale, QC * scale]


def split_blocks(image, block_width, block_height):
    image_segs = []
    for i in range(0, image.shape[0], block_height):
        for j in range(0, image.shape[1], block_width):
            image_segs.append(image[i:i+block_height, j:j+block_width])

    return image_segs


def combine_blocks(image_segs, image_width, image_height, sample_width, sample_height):
    x = image_height / sample_height
    y = image_width / sample_width

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

    # convert block to float
    block = block / 1.0

    # split the block into three channels
    block_channels = cv2.split(block)

    # take dct of each block channel
    for c in range(0, 3):
        block_channels[c]= cv2.dct(block_channels[c])

    # quantization
    for c in range(0, 3):
        block_channels[c] = block_channels[c] / quantization_table[c]
        block_channels[c] = block_channels[c].astype(int)

    # merge the block channels back into one ycrbr image
    block = cv2.merge(block_channels)

    return block


def decompress_block(block, quantization_table):

    # split the block into three channels
    block_channels = cv2.split(block)

    # inverse quantisation
    for c in range(0, 3):
        block_channels[c] = block_channels[c] * quantization_table[c]

    # take inverse dtc of block channel
    for c in range(0, 3):
        block_channels[c] = cv2.idct(block_channels[c])

    # merge the block channels back into one ycrbr image
    block = cv2.merge(block_channels)

    # shift range to be 0 - 255
    block = block + 128

    # truncate the block so that all pixels are within the range 0-255
    for i in range(0, block.shape[0]):
        for j in range(0, block.shape[1]):
            for c in range(0, 3):
                if block[i][j][c] > 255:
                    block[i][j][c] = 255
                elif block[i][j][c] < 0:
                    block[i][j][c] = 0

    return block


def write_to_bin(file_name, frequency_list, quality_factor, image_width, image_height, border_w, border_h, bin_str):

    # create the header for the binary file holding the width and height followed by the frequency list
    header_str = str(int(quality_factor)) + "_" + str(image_width) + "_" + str(image_height) + "_" + str(border_w) + "_" + str(border_h) + "_"
    print "Writing header string [", header_str, "]"
    for i in range(0, len(frequency_list)):
        header_str += str(frequency_list[i][0]) + "_" + str(frequency_list[i][1]) + "_"
    header_str += "\n"

    # work out how many bits will wasted by slicing into bytes
    bit_waste = (8 - (len(bin_str) % 8)) % 8

    # store bitwaste into the first byte of the code so that we know how much to ignore when decoding
    byte_chunks = ["{0:b}".format(bit_waste).zfill(8)]

    # cut the string into groups of 8 char making sure that any left over are prefixed by 0
    byte_chunks += [bin_str[i:i+8].zfill(8) for i in range(0, len(bin_str), 8)]

    # convert bytes into numbers
    byte_arr = []
    for b in byte_chunks:
        byte_arr.append(int(b, 2))

    # convert the numbers into actual bytes
    byte_array = bytearray(byte_arr)

    # the first line of the file will just be ascii for the width and height of the image
    bin_file = open(file_name, "wb")
    bin_file.write(header_str)
    bin_file.write(byte_array)


def read_from_bin(file_name):

    # read the file by appending bytes to array
    f = open(file_name, "rb")
    file_bytes = []
    header = ""
    done_with_header = False
    try:
        b = f.read(1)
        while b != "":
            if not done_with_header:
                if b == "\n":
                    done_with_header = True
                    b = f.read(1)
                else:
                    header += b
                    b = f.read(1)
            else:
                file_bytes.append(b)
                b = f.read(1)
    finally:
        f.close()

    print len(file_bytes)

    # reform the string from all bytes
    bit_string = "".join("{0:b}".format(ord(file_byte)).zfill(8) for file_byte in file_bytes)

    # remove any of the wasted bits that we dont want to be considered part of the huffman code
    bit_waste = "{0:b}".format(ord(file_bytes[0])).zfill(8)
    last_byte = bit_string[len(bit_string) - 8: len(bit_string)]
    last_byte = last_byte[int(bit_waste, 2): len(last_byte)]

    # remove the bit waste header
    bit_string = bit_string[8 : len(bit_string) - 8] + last_byte

    return header, bit_string


def rearrange_block(block_1d, block_size):

    # create a 3d array with the size 8 * 8 * 3
    block = []
    for i in range(0, block_size):
        temp_line = []
        for j in range(0, block_size):
            temp_channel = []
            for k in range(0, 3):
                temp_channel.append(0)
            temp_line.append(temp_channel)
        block.append(temp_line)

    # get the size of the block (should be 8)
    n = len(block)
    j = 0
    k = 0
    c = 0

    # for all of the left and bottom edge indexes for this n
    for i in range(0, n * 2 - 1):

        # for the current edge loop through the diagonal from this point
        x = j
        y = k
        while x >= 0 and y < len(block[0]):
            for l in range(0, 3):
                block[x][y][l] = block_1d[c]
                c += 1
            x -= 1
            y += 1

        # if i is still less than the block height keep iterating up
        if i < n - 1:
            j += 1

        # i reached the mid point, start moving the other way
        else:
            k += 1

    return block


def array_to_image(arr):

    # the size of the block
    n = len(arr)

    blank_image = np.zeros((n, n, 3), np.int8)

    for i in range(0, n):
        for j in range(0, n):
            for k in range(0, 3):
                blank_image[i][j][k] = arr[i][j][k]

    return blank_image


def compress_image(img, quality_factor, file_name):
    # define the size of a block
    dct_width = 8
    dct_height = 8

    # give the image a border so that it is divisible by 8 on both width and height
    border_w = img.shape[1] % dct_width
    border_h = img.shape[0] % dct_height
    img = cv2.copyMakeBorder(img, 0, border_h, 0, border_w, cv2.BORDER_CONSTANT)

    # store this for later
    image_width = img.shape[0]
    image_height = img.shape[1]

    # define the quality to compress, 0 is low quality 99 is high, then get quanization tables
    print "Getting quantisation tables for quality [", quality_factor, "]"
    Q = get_quantisation_channels(quality_factor)

    # convert from BGR to YCrBr colour space
    print "Converting to YCrCb"
    image_ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

    # divide the image into several 8*8*3 images
    print "Splitting image into 8 * 8 * 3 blocks"
    image_blocks = split_blocks(image_ycc, dct_width, dct_height)

    # compress blocks using the quantisation tables made eariler
    print "Compressing blocks with DCT"
    compressed_blocks = []
    for i in range(0, len(image_blocks)):
        compressed_blocks.append(compress_block(image_blocks[i], Q))

    # store all block values
    print "Getting all block values"
    block_channel_values = []
    for i in range(0, len(image_blocks)):
        block_channel_values.append(get_block_channel_values(compressed_blocks[i]))

    # join all values into a string for tree analysis
    print "Joining values to arr"

    all_block_values = []
    for i in range(0, len(block_channel_values)):
        all_block_values += block_channel_values[i]

    # get the frequency of each of the value so that we can build a tree
    print "Creating huffman tree"
    frequency_list = get_frequency_list(all_block_values)

    node_list = create_nodes(frequency_list)
    root_node = create_tree(node_list)

    # encode the big string we made
    print "Encoding str with huffman tree"
    bin_str = encode(root_node, all_block_values)

    # make a bit of a guess as to how big the file will be
    print "Current string size   [", len(bin_str) * 8, "(bits), ", len(bin_str), "(bytes), ", len(bin_str) / 1024, "(kb)]"
    print "Estimate on file size [", len(bin_str), "(bits), ", len(bin_str) / 8, "(bytes), ", (len(bin_str) / 8) / 1024, "(kb)]"

    # write the binary string to a file
    print "Writing encoded image to bin"
    write_to_bin(file_name, frequency_list, quality_factor, image_width, image_height, border_w, border_h, bin_str)


def decompress_image(file_name):

    dct_width = 8
    dct_height = 8

    # read in the file header and body
    header, bin_str = read_from_bin(file_name)
    print header

    # split the header into segments the first three being the quality factor, width and height
    h = header.split("_")
    quality_factor = int(h[0])
    image_width = int(h[1])
    image_height = int(h[2])
    border_w = int(h[3])
    border_h = int(h[4])

    print "Quality factor [", quality_factor, "]"
    print "Image width [", image_width, "]"
    print "Image height [", image_height, "]"
    print "Border width [", border_w, "]"
    print "Border height [", border_h, "]"

    Q = get_quantisation_channels(quality_factor)

    frequency_list = []
    for i in range(5, len(h) - 1, 2):
        frequency_list.append([int(h[i]), int(h[i+1])])

    node_list = create_nodes(frequency_list)
    root_node = create_tree(node_list)

    # get a directory of all of the values for faster comparisonss
    print "Getting directory from root node"
    decode_dir = get_decode_directory(root_node)

    # decode the binary string
    print "Using directory to decode binary string"
    decoded_image_arr = directory_decode(root_node, decode_dir, bin_str)

    decoded_blocks = []
    for i in range(0, len(decoded_image_arr), 192):
        decoded_blocks.append(array_to_image(rearrange_block(decoded_image_arr[i : i + 192], dct_width)))

    # decompress blocks using the same quantisation table
    print "Decompressing blocks with IDCT"
    decompressed_blocks = []
    for i in range(0, len(decoded_blocks)):
        decompressed_blocks.append(decompress_block(decoded_blocks[i], Q))

    # stitch together all of the decompressed blocks
    print "Stitching together all 8 * 8 * 3 blocks"
    image_ycc = combine_blocks(decompressed_blocks, image_width, image_height, dct_width, dct_height)

    # convert the new compressed image back into the BGR colour space
    print "Converting to BGR"
    image = cv2.cvtColor(image_ycc, cv2.COLOR_YCrCb2BGR)

    # trim off the border we crated earlier
    print "Removing any extra border"
    image = image[0: image_width - border_h, 0: image_height - border_w]

    return image


def get_size_in_mb(file_name):
    return int(os.path.getsize(file_name)) / 1024.0 / 1024.0


if __name__ == "__main__":

    image = "images/turtle.ppm"
    binary_file = "turtle.bin"
    mode = "d"
    comp_val = 5.0

    if mode == "c":
        print "Image file size [", get_size_in_mb(image), "MB]"

        start_time = time.time()
        compress_image(cv2.imread(image, 1), comp_val, binary_file)
        end_time = time.time()
        print "Compression time [", end_time - start_time, "]"

        print "Bin file size [", get_size_in_mb(binary_file), "MB]"

    else:
        start_time = time.time()
        image = decompress_image(binary_file)
        end_time = time.time()
        print "Decompression time [", end_time - start_time, "]"

        cv2.imshow("COMPRESSED", image)
        cv2.waitKey()