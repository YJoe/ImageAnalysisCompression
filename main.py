class Node:
    def __init__(self, left_node, right_node, value, frequency):
        self.left_node = left_node
        self.right_node = right_node
        self.value = value
        self.frequency = frequency

    def print_details(self):
        if self.left_node is not None:
            print "left      [", self.left_node.value, "]"
        if self.right_node is not None:
            print "right     [", self.right_node.value, "]"

        print "value     [", self.value, "]"
        print "frequency [", self.frequency, "]"


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
    for i in range(0, len(node_values)):
        if node_values[0].frequency >= root.frequency:
            left_node = Node(None, None, node_values[0].value, node_values[0].frequency)
            del node_values[0]
            root = Node(left_node, root, str(left_node.value + root.value), left_node.frequency + root.frequency)
            root, node_values = append_to_node_tree(root, node_values)
            break

    return root, node_values


def create_tree(node_values):

    roots = []

    # while there are still frequency values that haven't been appended to a tree
    while len(node_values) != 0:

        # if we can't make a pair then this single element should be in its own tree
        if len(node_values) == 1:
            root = node_values[0]
            del node_values[0]

        # we can make a pair so we will make a tree with them
        else:

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


def encode(root, regular_text):
    return "".join(encode_char(root, letter, "") for letter in regular_text)


def decode_char(root, code):
    if root.left_node is None and root.right_node is None:
        return root.value
    elif code == "":
        return None
    elif code[0] == "0":
        return decode_char(root.left_node, code[1:])
    else:
        return decode_char(root.right_node, code[1:])


def decode(root, code):
    output = ""

    while code != "":
        print "\nCode is now [", code, "]"
        for i in range(1, len(code) + 1):
            code_segment = code[0:i]
            print "\tsampling [", code_segment, "]"
            d = decode_char(root, code_segment)
            if d is not None:
                print "FOUND A BLOODY MATCH! [", code_segment, "] == [", d, "]"
                output += d
                code = code[i:]
                break

    return output


if __name__ == "__main__":

    # get some data to create a tree for
    text = "THIS IS SOME TEXT"
    frequency_list = get_frequency_list(text)
    node_list = create_nodes(frequency_list)

    print "All nodes"
    for node in node_list:
        node.print_details()

    root_node = create_tree(node_list)[0]

    print "\nRoot node of tree"
    root_node.print_details()

    print "\nEncoding [", text, "]"
    encoded_text = encode(root_node, text)
    print encoded_text

    print decode(root_node, encoded_text)


