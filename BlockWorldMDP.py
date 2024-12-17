import numpy as np

class BlocksWorldMDP:
    def __init__(self):
        self.colors = ["green", "yellow", "red"]
        self.num_piles = 4
        self.stacking_pile = 0  # The target pile for stacking
        self.num_actions = self.num_piles * (self.num_piles - 1)
        self.reward_target = 100
        self.reward_default = -1
        self.failure_prob = 0.0
        self.reset()

    def reset(self):
        """Initialize a random Blocks World state."""
        self.state = {
            "blocks": [
                {
                    "color": color,
                    "pile": np.random.randint(self.num_piles),
                    "height": -1,  # Placeholder, to be computed
                }
                for color in self.colors
            ]
        }
        self._update_heights()
        return self.state

    def _update_heights(self):
        """Update the height of blocks based on their pile."""
        pile_contents = {pile: [] for pile in range(self.num_piles)}
        for block in self.state["blocks"]:
            pile_contents[block["pile"]].append(block)
        
        for pile, blocks in pile_contents.items():
            # Sort blocks by height (lowest to highest)
            blocks.sort(key=lambda block: block["height"])
            for height, block in enumerate(blocks):
                block["height"] = height

    def _render_state(self):
        """Render the state as a string."""
        piles = {i: [] for i in range(self.num_piles)}
        for block in self.state["blocks"]:
            piles[block["pile"]].append((block["color"], block["height"]))  # Use color and height
        
        state_str = ""
        for pile in range(self.num_piles):
            if piles[pile]:
                pile_str = "-".join(f"{color[0]}({height})" for color, height in sorted(piles[pile], key=lambda x: x[1]))
            else:
                pile_str = "-"
            state_str += f"Pile {pile}: {pile_str}\n"
        print(state_str)

    def step(self, action):
        """
        Perform the given action and return (next_state, reward, done).
        
        Action is encoded as an integer:
            action = from_pile * num_piles + to_pile
        """
        from_pile = action // self.num_piles
        to_pile = action % self.num_piles

        print(f"Action: Move block from Pile {from_pile} to Pile {to_pile}")
        
        if from_pile == to_pile or np.random.rand() < self.failure_prob:
            # Invalid action or failed action
            print("Action failed or invalid. No change in state.")
            return self.state, self.reward_default, False

        # Find the top block in the from_pile
        moving_block = None
        highest_height = -1
        for block in self.state["blocks"]:
            if block["pile"] == from_pile and block["height"] > highest_height:
                moving_block = block
                highest_height = block["height"]

        if moving_block is None:
            # No block to move
            print("No block to move from the specified pile.")
            return self.state, self.reward_default, False

        # Move the block
        moving_block["pile"] = to_pile

        # Set the height of the moving block to be the top of the target pile
        new_pile_blocks = [block for block in self.state["blocks"] if block["pile"] == to_pile]
        moving_block["height"] = len(new_pile_blocks)  # Length determines the next height

        # Update heights to maintain consistency in all piles
        self._update_heights()

        # Compute reward
        stacking_pile_blocks = sorted(
            [block for block in self.state["blocks"] if block["pile"] == self.stacking_pile],
            key=lambda b: b["height"]
        )

        correct_order = ["green", "yellow", "red"]
        if len(stacking_pile_blocks) == len(self.colors) and \
           all(block["color"] == target_color for block, target_color in zip(stacking_pile_blocks, correct_order)):
            reward = self.reward_target
        else:
            reward = self.reward_default

        return self.state, reward, False

    def get_actions(self):
        """Return the list of all possible actions."""
        actions = []
        for from_pile in range(self.num_piles):
            for to_pile in range(self.num_piles):
                if from_pile != to_pile:
                    actions.append(from_pile * self.num_piles + to_pile)
        return actions

    def extract_transition_matrices(self):
        """
        Generate transition matrices for the MDP.
        Each action has a separate transition matrix.
        Rows represent current states, and columns represent next states.
        """
        num_states = self.num_piles ** len(self.colors)
        transition_matrices = np.zeros((self.num_actions, num_states, num_states))

        # Map states to indices
        state_to_index = {}
        index_to_state = {}

        state_counter = 0
        for piles in range(self.num_piles ** len(self.colors)):
            state = []
            temp = piles
            for _ in range(len(self.colors)):
                state.append(temp % self.num_piles)
                temp //= self.num_piles

            state_to_index[tuple(state)] = state_counter
            index_to_state[state_counter] = tuple(state)
            state_counter += 1

        # Populate transition matrices
        for action in range(self.num_actions):
            from_pile = action // self.num_piles
            to_pile = action % self.num_piles

            for state_index, state in index_to_state.items():
                if from_pile not in state:
                    # No block to move from this pile
                    transition_matrices[action, state_index, state_index] += 1
                    continue

                # Perform the action
                new_state = list(state)
                moving_block_index = state.index(from_pile)
                new_state[moving_block_index] = to_pile
                new_state_tuple = tuple(new_state)

                if new_state_tuple in state_to_index:
                    next_state_index = state_to_index[new_state_tuple]
                    transition_matrices[action, state_index, next_state_index] += (1 - self.failure_prob)
                    transition_matrices[action, state_index, state_index] += self.failure_prob

        return transition_matrices

# Example usage
if __name__ == "__main__":
    env = BlocksWorldMDP()
    state = env.reset()
    print("Initial State:")
    env._render_state()

    for _ in range(10):
        action = np.random.choice(env.get_actions())
        next_state, reward, done = env.step(action)
        print(f"Action: {action}, Reward: {reward}")
        print("Next State:")
        env._render_state()

    # Extract transition matrices
    transition_matrices = env.extract_transition_matrices()
    print("Transition Matrices Shape:", transition_matrices.shape)
