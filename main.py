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
    print "\t\t\tTrying to append to root [", root.value, "(", root.frequency, ")] something form [", ",".join(str(n.value) + "(" + str(n.frequency) + ")" for n in node_values), "]"
    for i in range(0, len(node_values)):
        if node_values[0].frequency >= root.frequency:
            print "\t\t\t\tThe next node frequency [", node_values[0].frequency, "] is >= [", root.frequency, "]"
            left_node = node_values[0]
            del node_values[0]
            root = Node(left_node, root, str(left_node.value + root.value), left_node.frequency + root.frequency)
            print "\t\t\t\t\tAppended left node [", left_node.value, "] to existing tree"
            return append_to_node_tree(root, node_values)

    # return the new root node and the node values that remain, if there are no node values then return an empty []
    return root, [] if node_values is None else node_values


def create_tree(node_values):

    print "\nCreating tree"

    roots = []

    # while there are still frequency values that haven't been appended to a tree
    while len(node_values) != 0:

        print "\tThere are still frequency values [", ",".join(str(n.value) + "(" + str(n.frequency) + ")" for n in node_values), "]"

        # if we can't make a pair then this single element should be in its own tree
        if len(node_values) == 1:
            print "\t\tWe can't make a pair though so placing [", node_values[0].value, "] in its own tree"
            root = node_values[0]
            del node_values[0]

        # we can make a pair so we will make a tree with them
        else:
            print "\t\tThere are at least two nodes left, creating pair"
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
        for i in range(1, len(code) + 1):
            code_segment = code[0:i]
            d = decode_char(root, code_segment)
            if d is not None:
                output += d
                code = code[i:]
                break

    return output


if __name__ == "__main__":

    # get some data to create a tree for
    text = "This is a message to encode, I'm going to make it long so that I can check that it works... I think it does now but who knows, maybe this will break it"
    frequency_list = get_frequency_list(text)
    node_list = create_nodes(frequency_list)

    print "All nodes"
    for node in node_list:
        print node.value, " ", node.frequency

    root_node = create_tree(node_list)[0]

    print "\nRoot node of tree"
    root_node.print_details()

    print "\nEncoding [", text, "]"
    encoded_text = encode(root_node, text)
    print encoded_text

    print "\nDecoding [", encoded_text, "]"
    print decode(root_node, encoded_text)

