def generate_label(state):
    """
    Generate a labeling dictionary for a given state of blocks.

    Args:
        state (tuple): A tuple of tuples representing the piles of blocks.

    Returns:
        dict: A dictionary with labels 'G', 'Y', or 'R' if conditions are met.
    """
    label = {}

    # Check if the green block is at the bottom of the last pile
    if state[3] and state[3][0] == 0:
        label['G'] = True
    else:
        label['G'] = False

    # Check if a yellow block is on top of a green block in any pile
    label['Y'] = any(
        0 in pile and pile[pile.index(0) + 1] == 1
        for pile in state
        if len(pile) > 1 and 0 in pile
    )

    # Check if a red block is on top of a green block in any pile
    label['R'] = any(
        0 in pile and pile[pile.index(0) + 1] == 2
        for pile in state
        if len(pile) > 1 and 0 in pile
    )

    return label

# Example usage
state = ((), (), (2, 1), (0,))
label_dict = generate_label(state)
print(label_dict)