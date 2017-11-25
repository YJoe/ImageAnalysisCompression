class Node:
    def __init__(self, left_node, right_node, value, frequency):
        self.left_node = left_node
        self.right_node = right_node
        self.value = value
        self.frequency = frequency


def get_frequency_list(data):
    frequency_data = []

    # for all of the data elements we have
    for i in range(0, len(text)):

        # assume we haven't found the element in the frequency list yet
        found_index = -1

        # look through all elements in the frequency list
        for j in range(0, len(frequency_data)):
            if text[i] == frequency_data[j][0]:
                # store the index we found the element
                found_index = j
                break

        # if we found the letter in the array add to the current count
        if found_index > -1:
            frequency_data[found_index][1] += 1

        # else we need to add a new element to the frequency list
        else:
            frequency_data.append([text[i], 1])

    # sort the frequency data by the frequency so that lowest is first
    frequency_data = sorted(frequency_data, key=lambda element: element[1])

    return frequency_data


def append_to_tree(root_node, frequency_values):
    print "\tChecking to see if we can append anything of ", frequency_values, "] to the current tree"
    for i in range(0, len(frequency_values)):
        print "\t\tCan we append [", frequency_values[0][1], "] when the current root freq is [", root_node.frequency, "] ?"
        if frequency_values[0][1] >= root_node.frequency:
            print "\t\t\tYes, creating new left node with [", frequency_values[0][0], "], right node will be the old root"
            left_node = Node(None, None, frequency_values[0][0], frequency_values[0][1])
            del frequency_values[0]
            root_node = Node(left_node, root_node, None, left_node.frequency + root_node.frequency)
            root_node, frequency_values = append_to_tree(root_node, frequency_values)
            break

    return root_node, frequency_values


if __name__ == "__main__":

    # get some data to create a tree for
    text = "this is some text"
    frequency_list = get_frequency_list(text)

    print "Frequency list"
    for f in frequency_list:
        print "\t", f[0], " ", f[1]

    tree_roots = []

    while len(frequency_list) != 0:

        # start a tree
        print "\nStarting a tree with [", frequency_list[1][0], "] and [", frequency_list[0][0], "]"
        left = Node(None, None, frequency_list[1][0], frequency_list[1][1])
        right = Node(None, None, frequency_list[0][0], frequency_list[0][1])
        root = Node(left, right, None, left.frequency + right.frequency)

        # remove the two nodes that are now in the tree
        del frequency_list[0]
        del frequency_list[0]

        # append to the tree
        root, frequency_list = append_to_tree(root, frequency_list)

        tree_roots.append(root)

    for tree_root in tree_roots:
        print tree_root.frequency
