class Node:
    def __init__(self, left_node, right_node, value, frequency):
        self.left_node = left_node
        self.right_node = right_node
        self.value = value
        self.frequency = frequency

    def print_details(self):
        print "left      [", self.left_node, "]"
        print "right     [", self.right_node, "]"
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
    print "\t\tChecking to see if we can append anything of [", "".join(n.value for n in node_values), "] to the current tree"
    for i in range(0, len(node_values)):
        print "\t\t\tCan we append [", node_values[0].frequency, "] when the current root freq is [", root.frequency, "] ?"
        if node_values[0].frequency >= root.frequency:
            print "\t\t\t\tYes, creating new left node with [", node_values[0].value, "], right node will be the old root [", root.value, "]"
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
            print "\n\tGoing to keep going, len is [", len(node_values), "]"

            # start a tree
            print "\tStarting a tree with [", node_values[1].value, "] and [", node_values[0].value, "]"
            left = node_values[1]
            right = node_values[0]
            root = Node(left, right, str(left.value + "" + right.value), left.frequency + right.frequency)

            # remove the two nodes that are now in the tree
            del node_values[0]
            del node_values[0]

            root, node_values = append_to_node_tree(root, node_values)

            print "\tLeft in the tree is [", ",".join(n.value for n in node_values), "]"

        roots.append(root)

    if len(roots) != 1:
        roots = create_tree(roots)

    return roots


if __name__ == "__main__":

    # get some data to create a tree for
    text = "this_is_some_text_to_encode"
    frequency_list = get_frequency_list(text)
    node_list = create_nodes(frequency_list)

    print "All nodes"
    for node in node_list:
        node.print_details()

    root_node = create_tree(node_list)[0]

    print "\nRoot node of tree"
    root_node.print_details()

