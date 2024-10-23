import config
from utils import bcolors
from semantic_predictor import (
    oracle_priority_predictor, 
    compute_total_generation_time, 
    cache_loading_time, 
    oracle_output_length_bucket_predictor,
    compute_predicted_total_generation_time
)

class Request:
    def __init__(self, args, user_request_id):
        # all methods in the class takes Request object as input

        self.args = args
        self.user_request_id = user_request_id
        # properties of the user request
        self.prompt_length = config.user_request_prompt_length[self.user_request_id]
        self.output_length = config.user_request_output_length[self.user_request_id]
        self.predicted_priority = oracle_priority_predictor(self.user_request_id)
        self.predicted_output_length = oracle_output_length_bucket_predictor(args, self.user_request_id)
        # it can have 1 unique parent
        self.parent = None
        # it can have multiple children unordered
        self.children = {}
        # record finished time and remaining time
        self.finished_computation_time = 0
        self.remaining_computation_time = compute_total_generation_time(args, self.user_request_id)
        self.predicted_remaining_computation_time = compute_predicted_total_generation_time(args, self.user_request_id)

    def set_parent(self, parent):
        self.parent = parent

    def add_child(self, child):
        self.children[child.user_request_id] = child

    def update_computation_time_normal_run(self, running_time):
        # update the prefill time needed, decoding time needed, and the total remaining time needed
        # after running another period of time, the actual running time
        # does not include saving cache loading cache time
        prefill_time, decoding_time, complete_running_time = self.remaining_computation_time
        if running_time > prefill_time:
            self.remaining_computation_time = (0, decoding_time - (running_time - prefill_time), complete_running_time-running_time)
        else:
            self.remaining_computation_time = (prefill_time - running_time, decoding_time, complete_running_time - running_time)

        self.finished_computation_time += running_time

    def update_predicted_computation_time_normal_run(self, running_time):
        # update the prefill time needed, decoding time needed, and the total remaining time needed
        # after running another period of time, the actual running time
        # does not include saving cache loading cache time
        prefill_time, decoding_time, complete_running_time = self.predicted_remaining_computation_time
        if running_time > prefill_time:
            self.predicted_remaining_computation_time = (0, decoding_time - (running_time - prefill_time), complete_running_time-running_time)
        else:
            self.predicted_remaining_computation_time = (prefill_time - running_time, decoding_time, complete_running_time - running_time)

    def update_computation_time_preempted(self):
        # update the time if preempted
        prefill_time, decoding_time, complete_running_time = self.remaining_computation_time
        cache_load_time = cache_loading_time(self.args, self.user_request_id, self.finished_computation_time)
        self.remaining_computation_time = (prefill_time, decoding_time, cache_load_time + complete_running_time)

    def update_predicted_computation_time_preempted(self):
        # update the time if preempted
        prefill_time, decoding_time, complete_running_time = self.predicted_remaining_computation_time
        cache_load_time = cache_loading_time(self.args, self.user_request_id, self.finished_computation_time)
        self.predicted_remaining_computation_time = (prefill_time, decoding_time, cache_load_time + complete_running_time)
            
    def print_out_features(self):
        print(bcolors.OKBLUE + f"User Request ID: {self.user_request_id}" + bcolors.ENDC)
        print(f"Priority: {self.predicted_priority}")
        print(f"Prompt Length: {self.prompt_length}")
        print(f"Output Length: {self.output_length}")
        print(f"Finished Computation Time: {self.finished_computation_time}")
        print(f"Remaining Computation Time: {self.remaining_computation_time}")

class PriorityQueue:
    def __init__(self, args):
        # all methods in this class take int (user_request_id) 
        # or list of int (pool_of_unordered_nodes) as input
        self.args = args
        self.root = Request(args, -1)
        self.nodes = {}

    def insert_node(self, user_request_id):
        node = Request(self.args, user_request_id)
        self.root.add_child(node)
        node.parent = self.root
        
        self.nodes[node.user_request_id] = node

    def remove_node(self, user_request_id):
        self.nodes[user_request_id].parent.children.pop(user_request_id, None)
        for child in self.nodes[user_request_id].children.values():
            child.set_parent(self.nodes[user_request_id].parent)
            self.nodes[user_request_id].parent.add_child(child)

        self.nodes.pop(user_request_id, None)
    
    def ordering(self, user_request_id_1, user_request_id_2):
        if self.nodes[user_request_id_1].predicted_priority < self.nodes[user_request_id_2].predicted_priority:
            return True 
        elif self.nodes[user_request_id_1].predicted_priority == self.nodes[user_request_id_2].predicted_priority:
            return self.nodes[user_request_id_1].predicted_remaining_computation_time[-1] < self.nodes[user_request_id_2].predicted_remaining_computation_time[-1]
        else:
            return False

    def add_dependency(self, user_request_id_1, user_request_id_2):
        # a comes before b
        self.nodes[user_request_id_2].parent.children.pop(user_request_id_2)
        self.nodes[user_request_id_1].add_child(self.nodes[user_request_id_2])
        self.nodes[user_request_id_2].set_parent(self.nodes[user_request_id_1])

        self.nodes[user_request_id_1] = self.nodes[user_request_id_1]
        self.nodes[user_request_id_2] = self.nodes[user_request_id_2]

    def get_highest_priority_request_id(self):
        # always make sure that the root has only one child
        # which means that we always know what's next to process to GPU
        if len(self.root.children) == 1:
            return list(self.root.children.keys())[0]
        else:
            unordered_nodes = list(self.root.children.keys())
            self.build_one_step_order(unordered_nodes)
            return list(self.root.children.keys())[0]
    
    def insert_node_in_correct_position(self, node, parent_node):
        # If node has a parent already, remove it from its parent's children
        if node.parent is not None and node.parent != parent_node:
            node.parent.children.pop(node.user_request_id, None)
        
        # Make a copy of the children list to avoid modification during iteration
        children = list(parent_node.children.values())
        for child in children:
            if self.ordering(node.user_request_id, child.user_request_id):
                # Node comes before the child
                # Remove child from parent_node's children
                parent_node.children.pop(child.user_request_id)
                # Set up new relationships
                node.add_child(child)
                child.set_parent(node)
                parent_node.add_child(node)
                node.set_parent(parent_node)
                return True
            elif self.ordering(child.user_request_id, node.user_request_id):
                # Child comes before the node
                if self.insert_node_in_correct_position(node, child):
                    return True
            else:
                # Nodes are not ordered; continue to next child
                continue
        # Node is not ordered with any children; add as a child of parent_node
        parent_node.add_child(node)
        node.set_parent(parent_node)
        return True

    def build_order(self, pool_of_unordered_nodes):
        """
        Total ordering of nodes in the pool with all other nodes in the tree
        """
        # worst case time complexity = O(n^2)
        for user_request_id in pool_of_unordered_nodes:
            node = self.nodes[user_request_id]
            self.insert_node_in_correct_position(node, self.root)

    def build_one_step_order(self, pool_of_unordered_nodes):
        """
        Given a list of node IDs that are immediate children of the root,
        find the node with the highest order among them and attach the other nodes
        (along with their subtrees) under it without further comparisons.
        """
        nodes = [self.nodes[node_id] for node_id in pool_of_unordered_nodes if node_id in self.nodes]

        # Ensure all nodes are immediate children of the root
        for node in nodes:
            assert node.parent == self.root, f"Node {node.user_request_id} is not an immediate child of the root."

        # Build the ordering among the nodes
        # Start with the first node
        ordered_nodes = [nodes[0]]

        for node in nodes[1:]:
            inserted = False
            # Compare node with each node in ordered_nodes to find its position
            for i, ordered_node in enumerate(ordered_nodes):
                if self.ordering(node.user_request_id, ordered_node.user_request_id):
                    # node comes before ordered_node
                    ordered_nodes.insert(i, node)
                    inserted = True
                    break
                elif self.ordering(ordered_node.user_request_id, node.user_request_id):
                    # node comes after ordered_node
                    continue
                else:
                    # No ordering; place node at the current position
                    continue
            if not inserted:
                # If not inserted, append at the end
                ordered_nodes.append(node)

        # Adjust the tree based on the ordering
        # The first node in ordered_nodes is the highest priority node
        highest_order_node = ordered_nodes[0]
        # Remove highest_order_node from root's children if necessary
        if highest_order_node.parent != self.root:
            highest_order_node.parent.children.pop(highest_order_node.user_request_id)
            highest_order_node.set_parent(self.root)
            self.root.add_child(highest_order_node)

        # Attach subsequent nodes under their appropriate parent based on ordering
        current_parent = highest_order_node
        for node in ordered_nodes[1:]:
            # Remove node from root's children
            node.parent.children.pop(node.user_request_id)
            # Set node's parent to current_parent
            node.set_parent(current_parent)
            # Add node to current_parent's children
            current_parent.add_child(node)
            # Update current_parent based on ordering
            current_parent = node
    
    def visualize(self, node, level=0):
        # Print the current node with indentation based on its level
        print(' ' * level * 4 + f"|-- {node.user_request_id}")
        
        # Recursively print each child at the next level
        for child in node.children.values():
            self.visualize(child, level + 1)
