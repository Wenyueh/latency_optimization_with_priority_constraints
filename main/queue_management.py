import config
from utils import bcolors, round_2
from heapq_utils import *
from semantic_predictor import (
    compute_prefill_time,
    compute_total_generation_time, 
    compute_generated_tokens,
    compute_predicted_total_generation_time,
    compute_optimal_prefill_cache_proportion,
    compute_optimal_decoding_cache_length,
    compute_output_time
)
import numpy as np
import operator
import argparse, asyncio
import random, math, time

class Request:
    def __init__(self, args, user_request_id, computed_priority_value, computed_output_length_bucket_value):
        # all methods in the class takes Request object as input

        self.args = args
        self.user_request_id = user_request_id
        # properties of the user request
        self.prompt_length = float('inf') if computed_output_length_bucket_value == float('inf') else config.user_request_prompt_length[self.user_request_id]
        self.output_length = float('inf') if computed_output_length_bucket_value == float('inf') else config.user_request_output_length[self.user_request_id]
        self.priority = float('inf') if computed_priority_value == float('inf') else config.user_request_priority[self.user_request_id]
        self.predicted_priority = float('inf') if computed_priority_value == float('inf') else computed_priority_value
        self.predicted_output_length = float('inf') if computed_output_length_bucket_value == float('inf') else max(1, computed_output_length_bucket_value * (args.max_output_length/args.length_bucket_num))

        # record finished time and remaining time
        self.finished_computation_time = 0
        self.remaining_computation_time = compute_total_generation_time(args, self)
        self.predicted_remaining_computation_time = compute_predicted_total_generation_time(args, self)
        self.waiting_time = 0

        self.optimal_prefill_cache_proportion = 0 if self.prompt_length == float('inf') else compute_optimal_prefill_cache_proportion(args, self.prompt_length)
        self.optimal_decoding_cache_length = 0 if self.prompt_length == float('inf') else compute_optimal_decoding_cache_length(args, self.prompt_length)
        self.CPU_prefilling_cache = [] # with respect to prompt length
        self.CPU_prefilling_loaded_cache = []
        self.prefill_cache_loading_time = 0

        self.CPU_decoding_cache = []
        self.CPU_decoding_loaded_cache = []
        self.decoding_cache_loading_time = 0

        # the previous request, if preempted, need to spend some time to save its cache
        self.previous_save_time = 0
        self.total_time = compute_total_generation_time(args, self)

    def reinitialize(self):
        # reinitialize the request
        self.finished_computation_time = 0
        self.remaining_computation_time = compute_total_generation_time(self.args, self)
        self.predicted_remaining_computation_time = compute_predicted_total_generation_time(self.args, self)

        self.CPU_prefilling_cache = []
        self.CPU_prefilling_loaded_cache = []
        self.prefill_cache_loading_time = 0

        self.CPU_decoding_cache = []
        self.CPU_decoding_loaded_cache = []
        self.decoding_cache_loading_time = 0

    def swap_or_delete_update(self, remaining_tokens_on_GPU, total_tokens_on_cache):
        prompt_length = self.prompt_length
        prefill_time, _, _ = self.remaining_computation_time

        if remaining_tokens_on_GPU > prompt_length:
            # don't need to update prefilling time
            update_prefill = False 
            on_GPU_decoded_token_number = remaining_tokens_on_GPU - prompt_length
        else:
            update_prefill = True
            on_GPU_decoded_token_number = 0
            on_GPU_prefilled_token_number = remaining_tokens_on_GPU
        
        # consider decoding part
        optimal_decoding_cache_length = compute_optimal_decoding_cache_length(self.args, prompt_length)
        total_generated_tokens = total_tokens_on_cache - prompt_length if total_tokens_on_cache > prompt_length else 0
        # if more than 0 tokens need to be recomputed
        if total_generated_tokens > optimal_decoding_cache_length:
            # the number of tokens to be recomputed
            recompute_tokens = total_generated_tokens - max(on_GPU_decoded_token_number, optimal_decoding_cache_length)
            # the recomputation time
            recomputation_time = compute_output_time(self.args, prompt_length + on_GPU_decoded_token_number, recompute_tokens) 
            self.remaining_computation_time[1] += recomputation_time
            self.remaining_computation_time[-1] += recomputation_time
            self.predicted_remaining_computation_time[1] += recomputation_time
            self.predicted_remaining_computation_time[-1] += recomputation_time
            self.finished_computation_time -= recomputation_time

        # the tokens to be cached
        cache_max_token = max(optimal_decoding_cache_length, total_generated_tokens)
        # generated tokens are indexed from 0 -> total_generated_tokens
        cached_tokens = list(range(max(on_GPU_decoded_token_number,optimal_decoding_cache_length), cache_max_token))
        self.CPU_decoding_cache += cached_tokens
        # there might be some duplicate tokens in the cache if directly appending
        self.CPU_decoding_cache = list(set(self.CPU_decoding_cache))
        self.decoding_cache_loading_time = len(self.CPU_decoding_cache) * self.args.cache_loading_speed

        if update_prefill:
            # consider prefilling part
            finished_prefilling_time = compute_prefill_time(self.args, prompt_length) - prefill_time
            if compute_optimal_prefill_cache_proportion(self.args, prompt_length) == 0:
                updated_prefill_time = round_2((self.prompt_length ** 2)*self.args.prefill_speed_coefficient1 + self.args.prefill_speed_coefficient2 * (self.prompt_length - on_GPU_prefilled_token_number))
                self.remaining_computation_time[0] = updated_prefill_time
                self.remaining_computation_time[-1] = self.remaining_computation_time[-1] - prefill_time + updated_prefill_time
                self.predicted_remaining_computation_time[0] = updated_prefill_time
                self.predicted_remaining_computation_time[-1] = self.predicted_remaining_computation_time[-1] - prefill_time + updated_prefill_time
                self.finished_computation_time -= finished_prefilling_time
                self.CPU_prefilling_cache = []
                self.prefill_cache_loading_time = 0
            else:
                self.CPU_prefilling_cache = list(range(on_GPU_prefilled_token_number, total_tokens_on_cache))
                self.prefill_cache_loading_time = len(self.CPU_prefilling_cache) * self.args.cache_loading_speed

    def update_normal_prefilling_iteration(self):
        self.CPU_prefilling_loaded_cache = self.CPU_prefilling_cache
        self.CPU_prefilling_cache = []
        self.prefill_cache_loading_time = 0

        self.remaining_computation_time[-1] -= self.remaining_computation_time[0]
        self.predicted_remaining_computation_time[-1] -= self.predicted_remaining_computation_time[0]
        self.remaining_computation_time[0] = 0
        self.predicted_remaining_computation_time[0] = 0

    def update_normal_decoding_iteration(self, MaxHeap_Memory):
        self.CPU_prefilling_cache = []
        self.CPU_prefilling_loaded_cache = []
        self.prefill_cache_loading_time = 0

        self.CPU_decoding_loaded_cache = self.CPU_decoding_cache
        self.CPU_decoding_cache = []    
        self.decoding_cache_loading_time = 0

        decoded_cache_length = max(self.CPU_decoding_loaded_cache) if self.CPU_decoding_loaded_cache != [] else max(0, MaxHeap_Memory.request_id2tokens[self.user_request_id]-1 - self.prompt_length)
        computing_time = compute_output_time(self.args, self.prompt_length + decoded_cache_length, 1)
        self.remaining_computation_time[1] -= computing_time
        self.predicted_remaining_computation_time[1] -= computing_time
        self.remaining_computation_time[-1] -= computing_time
        self.predicted_remaining_computation_time[-1] -= computing_time
        self.remaining_computation_time[0] = 0
        self.predicted_remaining_computation_time[0] = 0

    def update_waiting_time(self, waiting_time):
        self.waiting_time += waiting_time

    def print_out_features(self):
        print(bcolors.OKBLUE + f"User Request ID: {self.user_request_id}" + bcolors.ENDC)
        print(f"Priority: {self.predicted_priority}")
        print(f"Prompt Length: {self.prompt_length}")
        print(f"Output Length: {self.output_length}")
        print(f"Predicted Output Length: {self.predicted_output_length}")
        print(f"Finished Computation Time: {self.finished_computation_time}")
        print(f"Predicted Remaining Computation Time: {self.predicted_remaining_computation_time}")
        print(f"Remaining Computation Time: {self.remaining_computation_time}")

# write a heap with two priorities: priority and remaining computation time
# using python heapq

# don't use three dimentional because the order of the user requests is not fixed at any time
class TwoDimensionalPriorityQueue:
    def __init__(self):
        self.heap = []

    def __len__(self):
        return len(self.heap)

    def delete(self, item):
        # item is a Request object, we need to find it in the heap
        item_list = [r for r in self.heap if item == r[2]]
        if len(item_list) != 1:
            assert len(item_list) == 1, "Item not found in the heap"
        item = item_list[0]
        # item is a Request object, we need to extend it to a tuple (priority1, priority2, item)
        # and then do the deletion
        index = self.heap.index(item)
        # If the item is the last element, just pop it
        if index == len(self.heap) - 1:
            return self.heap.pop()
        # Swap with the last element
        self.heap[index], self.heap[-1] = self.heap[-1], self.heap[index]
        self.heap.pop()  # actually remove it

        # Restore the heap property:
        # Try heapify_down first; if that doesn't fix, we do a heapify_up
        heapify(self.heap)

        # print in red color
        print(bcolors.FAIL + f"Deleted item: {item[2].user_request_id}" + bcolors.ENDC)

    def push(self, item, priority1, priority2):
        # semantic priority, remaining computation time
        # Use a tuple (priority1, priority2, item) to maintain the heap property
        heappush(self.heap, (priority1, priority2, item))

    def pop(self):
        # Pop the item with the highest priority (smallest priority1, then smallest priority2)
        return heappop(self.heap)[2]

    def peek(self):
        # Peek at the item with the highest priority without popping it
        return self.heap[0][2]

    def is_empty(self):
        return len(self.heap) == 0

class PriorityQueue:
    def __init__(self, args):
        # all methods in this class take int (user_request_id) 
        # or list of int (pool_of_unordered_nodes) as input
        self.args = args
        self.unsorted_nodes = []
        self.first_unsorted_node_idx = -1
        self.two_dim_priority_queue = TwoDimensionalPriorityQueue()
        self.is_GPU_job_completed = False

    def return_all_requests(self):
        return self.unsorted_nodes + [q[2] for q in self.two_dim_priority_queue.heap]

    def add_unsorted_node(self, user_request):
        self.unsorted_nodes.append(user_request)
        if self.first_unsorted_node_idx < 0:
            if len(self.unsorted_nodes) == 1:
                self.first_unsorted_node_idx = 0
            else:
                # max unsorted node has been fetched before adding the new element,
                # we need to recompute the max unsorted node index from scratch after adding the new element
                self.find_max_unsorted_node_idx()
                return
            
        if user_request.predicted_priority < self.unsorted_nodes[self.first_unsorted_node_idx].predicted_priority:
            self.first_unsorted_node_idx = len(self.unsorted_nodes) - 1
        elif user_request.predicted_priority == self.unsorted_nodes[self.first_unsorted_node_idx].predicted_priority:
            if (user_request.predicted_remaining_computation_time[-1] \
                + user_request.prefill_cache_loading_time \
                + user_request.decoding_cache_loading_time) \
                <= (self.unsorted_nodes[self.first_unsorted_node_idx].predicted_remaining_computation_time[-1] \
                + self.unsorted_nodes[self.first_unsorted_node_idx].prefill_cache_loading_time \
                + self.unsorted_nodes[self.first_unsorted_node_idx].decoding_cache_loading_time):
                self.first_unsorted_node_idx = len(self.unsorted_nodes) - 1

    def find_max_unsorted_node_idx(self):
        assert self.first_unsorted_node_idx < 0
        self.first_unsorted_node_idx = -1

        highest_priority = self.unsorted_nodes[self.first_unsorted_node_idx].predicted_priority
        smallest_remaining_time = self.unsorted_nodes[self.first_unsorted_node_idx].predicted_remaining_computation_time[-1] + self.unsorted_nodes[self.first_unsorted_node_idx].prefill_cache_loading_time + self.unsorted_nodes[self.first_unsorted_node_idx].decoding_cache_loading_time
        for idx, user_request in enumerate(self.unsorted_nodes):
            if user_request.predicted_priority < highest_priority:
                self.first_unsorted_node_idx = idx
                highest_priority = user_request.predicted_priority
                smallest_remaining_time = user_request.predicted_remaining_computation_time[-1] + user_request.prefill_cache_loading_time + user_request.decoding_cache_loading_time
            elif user_request.predicted_priority == highest_priority:
                if (user_request.predicted_remaining_computation_time[-1] + user_request.prefill_cache_loading_time + user_request.decoding_cache_loading_time < smallest_remaining_time):
                    self.first_unsorted_node_idx = idx
                    highest_priority = user_request.predicted_priority
                    smallest_remaining_time = user_request.predicted_remaining_computation_time[-1] + user_request.prefill_cache_loading_time + user_request.decoding_cache_loading_time
    
    def find_top_k_nodes_in_unsorted_list(self, unsorted_nodes, k):
        # find the top k nodes in the unsorted list
        top_k_nodes = []
        if len(unsorted_nodes) == 0:
            return top_k_nodes
        # find the first unsorted node index
        if self.first_unsorted_node_idx < 0:
            self.find_max_unsorted_node_idx()
        # find the top k nodes
        for _ in range(k):
            if self.first_unsorted_node_idx < 0:
                break
            top_k_nodes.append(unsorted_nodes[self.first_unsorted_node_idx])
            unsorted_nodes[self.first_unsorted_node_idx] = Request(self.args, -1, float('inf'), float('inf'))
            self.find_max_unsorted_node_idx()
        return top_k_nodes
    
    def fetch_next_node(self):
        # assert len(self.root.children) == 1, "Root should have only one child."       
        if self.two_dim_priority_queue.is_empty():
            next_sorted_node = Request(self.args, -1, float('inf'), float('inf'))
        else:
            next_sorted_node = self.two_dim_priority_queue.peek()

        if len(self.unsorted_nodes) > 0:
            if self.first_unsorted_node_idx < 0:
                # re-compute the first unsorted node index if it is not valid
                self.find_max_unsorted_node_idx()

            next_unsorted_node = self.unsorted_nodes[self.first_unsorted_node_idx]
            if (next_unsorted_node.predicted_priority < next_sorted_node.predicted_priority):
                # lazy deletion by replacing the node with a dummy node
                self.unsorted_nodes[self.first_unsorted_node_idx] = Request(self.args, -1, float('inf'), float('inf'))
                self.first_unsorted_node_idx = -1
                return next_unsorted_node
            elif (next_unsorted_node.predicted_priority == next_sorted_node.predicted_priority):
                if (
                        next_unsorted_node.predicted_remaining_computation_time[-1] 
                        + next_unsorted_node.prefill_cache_loading_time  
                        + next_unsorted_node.decoding_cache_loading_time 
                        <= next_sorted_node.predicted_remaining_computation_time[-1]
                        + next_sorted_node.prefill_cache_loading_time 
                        + next_sorted_node.decoding_cache_loading_time
                    ):
                    # lazy deletion by replacing the node with a dummy node
                    self.unsorted_nodes[self.first_unsorted_node_idx] = Request(self.args, -1, float('inf'), float('inf'))
                    self.first_unsorted_node_idx = -1
                    return next_unsorted_node

        self.two_dim_priority_queue.pop()
        return next_sorted_node
    
    def fetch_next_k_nodes(self, k, require_decode=False, max_prefilling_time=float('inf')):
        all_nodes = []
        insert_back_nodes = []

        if require_decode:
            while len(all_nodes) < k:
                next_node = self.fetch_next_node()
                if next_node.user_request_id == -1:
                    break
                if next_node.predicted_remaining_computation_time[0] <= 0:
                    all_nodes.append(next_node)
                else:
                    insert_back_nodes.append(next_node)

                if len(self.unsorted_nodes) + len(self.two_dim_priority_queue.heap) == 0:
                    break
        else:
            while len(all_nodes) < k:
                next_node = self.fetch_next_node()
                if next_node.user_request_id == -1:
                    break
                                
                if next_node.predicted_remaining_computation_time[0] <= max_prefilling_time:
                    all_nodes.append(next_node)
                else:
                    all_nodes.append(next_node)
                    #insert_back_nodes.append(next_node)

                max_prefilling_time = min(next_node.predicted_remaining_computation_time[0], max_prefilling_time)

                if len(self.unsorted_nodes) + len(self.two_dim_priority_queue.heap) == 0:
                    break

        for one_node in insert_back_nodes:
            self.two_dim_priority_queue.push(one_node, one_node.predicted_priority, one_node.predicted_remaining_computation_time[-1] + one_node.prefill_cache_loading_time + one_node.decoding_cache_loading_time)

        return all_nodes

    async def incremental_update(self):
        while len(self.unsorted_nodes) > 0:
            next_unsorted_node = self.unsorted_nodes.pop(0)
            self.first_unsorted_node_idx -= 1
            # delete the dummy node
            if next_unsorted_node.predicted_priority < float('inf'):
                self.two_dim_priority_queue.push(next_unsorted_node, next_unsorted_node.predicted_priority, next_unsorted_node.predicted_remaining_computation_time[-1]+ next_unsorted_node.prefill_cache_loading_time + next_unsorted_node.decoding_cache_loading_time)

    def update_waiting_time(self, waiting_time):
        for request in self.unsorted_nodes:
            if request.predicted_priority < float('inf'):
                request.update_waiting_time(waiting_time)
        # go through all the nodes in the heap
        for idx in range(len(self.two_dim_priority_queue.heap)):
            self.two_dim_priority_queue.heap[idx][2].waiting_time += waiting_time

    def visualize(self):
        def visualize_unsorted_nodes(unsorted_nodes):
            print(bcolors.OKGREEN + "Unsorted Nodes:" + bcolors.ENDC)
            for user_request in unsorted_nodes:
                print(bcolors.OKBLUE + f"User Request ID: {user_request.user_request_id}, Predicted Priority: {user_request.predicted_priority}, Predicted Computation Time: {user_request.predicted_remaining_computation_time[-1] + user_request.prefill_cache_loading_time + user_request.decoding_cache_loading_time}" + bcolors.ENDC)

        # visualize a heap 
        def print_heap_tree(heap, index=0, indent="", branch="Root: "):
            """
            Recursively prints a heap as a tree.
            
            :param heap: List representing the heap
            :param index: Current index within 'heap'
            :param indent: Current indentation (used internally)
            :param branch: Label of the current branch (Root, L---, R---)
            """
            if index < len(heap):
                # Print the current node
                print(indent + branch + str((heap[index][2].user_request_id, heap[index][2].predicted_priority, heap[index][2].predicted_remaining_computation_time[-1] + heap[index][2].prefill_cache_loading_time + heap[index][2].decoding_cache_loading_time)))
                
                # Prepare for children
                left_index = 2 * index + 1
                right_index = 2 * index + 2
                
                # Increase indentation for children
                child_indent = indent + "    "
                
                # Recursively print left and right subtrees
                print_heap_tree(heap, left_index, child_indent, "L--- ")
                print_heap_tree(heap, right_index, child_indent, "R--- ")

        print(bcolors.OKGREEN + "Data Management for Waiting Requests (MinHeap):" + bcolors.ENDC)
        print_heap_tree(self.two_dim_priority_queue.heap)
        visualize_unsorted_nodes(self.unsorted_nodes)

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--seed', type=int, default=0)
#     parser.add_argument('--user_request_gap', type=float, default=0.5)
#     parser.add_argument('--user_request_num', type=int, default=10)
#     parser.add_argument('--ranks', type=int, default=10)
#     parser.add_argument('--length_bucket_num', type=int, default=20)
#     parser.add_argument('--max_prompt_length', type=int, default=20)
#     parser.add_argument('--max_output_length', type=int, default=100)

#     parser.add_argument('--token_decoding_speed', type=float, default=0.05)
#     parser.add_argument('--prefill_speed', type=float, default=0.05)
#     parser.add_argument('--cache_loading_speed', type=float, default=0.01, help='cache loading speed per token')
#     parser.add_argument('--cache_saving_speed', type=float, default=0.01, help='cache saving speed per token')
#     args = parser.parse_args()

#     def initialize_user_request(args):
#         user_request_priority = [random.choice(list(range(args.ranks))) for _ in range(args.user_request_num)]
#         user_request_prompt_length = [random.choice(list(range(1, args.max_prompt_length))) for _ in range(args.user_request_num)]
#         user_request_output_length = [random.choice(list(range(1, args.max_output_length))) for user_request in user_request_prompt_length]
#         return user_request_priority, user_request_prompt_length, user_request_output_length
    
#     random.seed(args.seed)
#     np.random.seed(args.seed)

#     config.user_request_priority, config.user_request_prompt_length, config.user_request_output_length = initialize_user_request(args)

#     queue = PriorityQueue(args)
#     node = Request(args, 0, 3, 2)
#     queue.insert_node(node)
#     node = Request(args, 1, 3, 3)
#     queue.insert_node(node)
#     node = Request(args, 2, 2, 4)
#     queue.insert_node(node)
#     node = Request(args, 3, 4, 5)
#     queue.insert_node(node)
#     node = Request(args, 4, 5, 6)
#     queue.insert_node(node)
#     node = Request(args, 5, 6, 7)
#     queue.insert_node(node)

#     queue.add_dependency(0, 1)
#     queue.add_dependency(2, 0)
#     queue.add_dependency(3, 4)
#     queue.add_dependency(3, 5)
#     queue.add_dependency(3, 2)

#     # queue.compute_first_node()
#     # queue.fetch_next_node().print_out_features()

#     # queue.build_order(list(range(10)))
#     # queue.remove_node(5)

#     # for user_request_id in range(10, 20):
#     #     queue.insert_node(user_request_id)

#     # print(queue.get_highest_priority_request_id())

#     # queue.remove_node(2)

#     # print(queue.get_highest_priority_request_id())

#     queue.visualize(queue.root)