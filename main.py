import numpy
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


def image_to_string(image):

    height, width, depth = image.shape
    flat = image.flatten()

    image_string = "[" + str(width) + "," + str(height) + "," + str(depth) + "]"

    for i in range(0, len(flat) / 3, 3):
        image_string += "[" + ",".join(str(flat[i + j]) for j in range(0, 3)) + "]"

    return image_string


def string_to_image(string):
    arr = re.findall("\[(.*?)\]", string)

    # the first set is [width, height, depth] of the image
    width, height, depth = arr[0].split(",")
    print width, " ", height, " ", depth


def dct_level_off(image, sub):
    return image - [sub, sub, sub]


if __name__ == "__main2__":
    img = cv2.imread('images/planet.jpg', 1)
    cv2.imshow("", img)
    cv2.waitKey()
    cv2.destroyAllWindows()

    print img[200][200]
    img = dct_level_off(img, 128)
    print img[200][200]


if __name__ == "__main__":

    img = cv2.imread('images/turtle.jpg', 1)
    cv2.imshow("", img)
    cv2.waitKey()
    cv2.destroyAllWindows()

    start_time = time.time()
    text = image_to_string(img)
    frequency_list = get_frequency_list(text)
    node_list = create_nodes(frequency_list)
    root_node = create_tree(node_list)[0]
    #print_network(root_node, "")
    end_time = time.time()
    print "Tree creation time [", end_time - start_time, "]"

    print "\nEncoding string"
    start_time = time.time()
    encoded_text = encode(root_node, text)
    end_time = time.time()
    print "Encode time [", end_time - start_time, "]"

    print "\nDecoding string"
    start_time = time.time()
    decoded_text = decode(root_node, encoded_text)
    end_time = time.time()
    print "Decode time [", end_time - start_time, "]"

    print "\nString to image"
    start_time = time.time()
    string_to_image(decoded_text)
    end_time = time.time()
    print "String to image time [", end_time - start_time, "]"

    print "\nDone"



