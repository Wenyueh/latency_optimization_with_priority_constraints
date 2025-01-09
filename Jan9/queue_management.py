import config
from utils import bcolors, round_2
from heapq_utils import *
from semantic_predictor import (
    compute_prefill_time,
    compute_total_generation_time, 
    compute_generated_tokens,
    compute_predicted_total_generation_time,
    compute_optimal_prefill_length,
    compute_optimal_decoding_length
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
        self.predicted_output_length = float('inf') if computed_output_length_bucket_value == float('inf') else computed_output_length_bucket_value * (args.max_output_length/args.length_bucket_num)

        # record finished time and remaining time
        self.finished_computation_time = 0
        self.remaining_computation_time = compute_total_generation_time(args, self)
        self.predicted_remaining_computation_time = compute_predicted_total_generation_time(args, self)
        self.waiting_time = 0

        self.optimal_prefill_cache_proportion = 0 if self.prompt_length == float('inf') else compute_optimal_prefill_length(args, self.prompt_length)/self.prompt_length
        self.optimal_decoding_cache_length = 0 if self.prompt_length == float('inf') else compute_optimal_decoding_length(args, computed_output_length_bucket_value * (args.max_output_length/args.length_bucket_num))
        self.prefill_cache_proportion = 0 # with respect to prompt length
        self.prefill_cache_loading_time = 0
        self.prefill_cache_loaded_proportion = 0 # what proportion of the cache is loaded with respect to prompt length

        self.decoding_cache_length = 0
        self.decoding_cache_loading_time = 0
        self.decoding_cache_loaded_length = 0 # how many tokens are loaded

        # the previous request, if preempted, need to spend some time to save its cache
        self.previous_save_time = 0

    ## assuming that the request can be run in the next interval
    ## don't consider the preemption condition
    ## and thus we don't compute the caching or loading time
    def update_computation_time_normal_run(self, running_time):
        if running_time >= self.previous_save_time:
            running_time -= self.previous_save_time
            self.previous_save_time = 0
        else:
            self.previous_save_time = max(0, self.previous_save_time - running_time)
            return 

        # remaining loading time of cache
        prefill_cache_loading_time_remained = round_2(self.prompt_length * self.args.cache_loading_speed * (self.prefill_cache_proportion-self.prefill_cache_loaded_proportion))
        decoding_cache_loading_time_remained = round_2(self.args.cache_loading_speed * (self.decoding_cache_length-self.decoding_cache_loaded_length))

        # if the running time is not enough to load the prefilling cache, then we don't need to update the time at all
        if running_time <= prefill_cache_loading_time_remained:
            self.prefill_cache_loaded_proportion += round_2(running_time / (self.prompt_length * self.args.cache_loading_speed))
        # if the running time is enough to load the prefilling cache, we check whether it is enough to finish prefilling, if not, we update the time needed to finish prefilling
        elif running_time > prefill_cache_loading_time_remained and running_time <= prefill_cache_loading_time_remained + self.remaining_computation_time[0]:
            self.prefill_cache_loaded_proportion = self.prefill_cache_proportion
            time_left_for_prefilling = running_time - prefill_cache_loading_time_remained
            self.remaining_computation_time[0] -= time_left_for_prefilling
            self.predicted_remaining_computation_time[0] = max(0, self.predicted_remaining_computation_time[0]- time_left_for_prefilling)
            self.remaining_computation_time[-1] -= time_left_for_prefilling
            self.predicted_remaining_computation_time[-1] = max(0, self.predicted_remaining_computation_time[-1]-time_left_for_prefilling)
            self.finished_computation_time += time_left_for_prefilling
        # if the running time is enough to load the prefilling cache and prefilling computation
        # then we check whether it is enough to finish loading decoding time, if not, we update the time needed to finish prefilling
        elif running_time > prefill_cache_loading_time_remained + self.remaining_computation_time[0] and running_time <= prefill_cache_loading_time_remained + self.remaining_computation_time[0] + decoding_cache_loading_time_remained:
            time_left_for_loading_decoding_cache = running_time - prefill_cache_loading_time_remained - self.remaining_computation_time[0]
            self.decoding_cache_loaded_length += int(time_left_for_loading_decoding_cache/self.args.cache_loading_speed)

            self.prefill_cache_loaded_proportion = self.prefill_cache_proportion
            self.remaining_computation_time[-1] -= self.remaining_computation_time[0]
            self.predicted_remaining_computation_time[-1] = max(0, self.predicted_remaining_computation_time[-1] - self.remaining_computation_time[0])
            self.finished_computation_time += self.remaining_computation_time[0]
            self.remaining_computation_time[0] = 0
            self.predicted_remaining_computation_time[0] = 0
        # if the running time covers prefilling cache + prefilling computation, then we check whether it is enough to finish decoding cache loading, if not, we just update prefilling time
        else:
            assert running_time > prefill_cache_loading_time_remained + self.remaining_computation_time[0] + decoding_cache_loading_time_remained
            self.prefill_cache_loaded_proportion = self.prefill_cache_proportion
            self.decoding_cache_loaded_length = self.decoding_cache_length

            time_left_for_decoding = running_time - prefill_cache_loading_time_remained - self.remaining_computation_time[0] - decoding_cache_loading_time_remained
            self.finished_computation_time += time_left_for_decoding + self.remaining_computation_time[0]
            self.remaining_computation_time = [0, self.remaining_computation_time[1] - time_left_for_decoding, self.remaining_computation_time[-1] - time_left_for_decoding - self.remaining_computation_time[0]]
            self.predicted_remaining_computation_time = [0, max(0,self.predicted_remaining_computation_time[1] - time_left_for_decoding), max(0, self.predicted_remaining_computation_time[-1] - time_left_for_decoding - self.predicted_remaining_computation_time[0])]

        self.finished_computation_time = round_2(self.finished_computation_time)
        self.remaining_computation_time = round_2(self.remaining_computation_time)
        self.predicted_remaining_computation_time = round_2(self.predicted_remaining_computation_time)
        if self.remaining_computation_time[1] == 0:
            self.predicted_remaining_computation_time[1] = 0
        if self.remaining_computation_time[-1] == 0:
            self.predicted_remaining_computation_time[-1] = 0

    def update_waiting_time(self, waiting_time):
        self.waiting_time += waiting_time

    def print_out_features(self):
        print(bcolors.OKBLUE + f"User Request ID: {self.user_request_id}" + bcolors.ENDC)
        print(f"Priority: {self.predicted_priority}")
        print(f"Prompt Length: {self.prompt_length}")
        print(f"Output Length: {self.output_length}")
        print(f"Finished Computation Time: {self.finished_computation_time}")
        print(f"Remaining Computation Time: {self.remaining_computation_time}")

# write a heap with two priorities: priority and remaining computation time
# using python heapq

class TwoDimensionalPriorityQueue:
    def __init__(self):
        self.heap = []

    def push(self, item, priority1, priority2):
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

    def add_unsorted_node(self, user_request):
        self.unsorted_nodes.append(user_request)
        # TODO: use the total remaining time of the user requests instead of 'predicted_remaining_computation_time'
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
            return
        elif user_request.predicted_priority == self.unsorted_nodes[self.first_unsorted_node_idx].predicted_priority:
            if (user_request.predicted_remaining_computation_time[-1] \
                + user_request.prefill_cache_loading_time \
                + user_request.decoding_cache_loading_time) \
                <= (self.unsorted_nodes[self.first_unsorted_node_idx].predicted_remaining_computation_time[-1] \
                + self.unsorted_nodes[self.first_unsorted_node_idx].prefill_cache_loading_time \
                + self.unsorted_nodes[self.first_unsorted_node_idx].decoding_cache_loading_time):
                self.first_unsorted_node_idx = len(self.unsorted_nodes) - 1
                return

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

            # TODO: use the total remaining time of the user requests instead of 'predicted_remaining_computation_time'
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

    async def incremental_update(self):
        self.is_GPU_job_completed = False
        # TODO: add the 'is_GPU_job_completed' flag to the PriorityQueue class and include it in the following condition to ensure atomic operation
        # TODO: the 'is_GPU_job_completed' flag should be set to True when the GPU job is completed, and False at the beginning of the next GPU job
        while len(self.unsorted_nodes) > 0 and not self.is_GPU_job_completed:
            next_unsorted_node = self.unsorted_nodes.pop(0)
            self.first_unsorted_node_idx -= 1
            # delete the dummy node
            # TODO: use the total remaining time of the user requests instead of 'predicted_remaining_computation_time'
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

        print(bcolors.OKGREEN + "Min Heap:" + bcolors.ENDC)
        print_heap_tree(self.two_dim_priority_queue.heap)
        visualize_unsorted_nodes(self.unsorted_nodes)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--user_request_gap', type=float, default=0.5)
    parser.add_argument('--user_request_num', type=int, default=10)
    parser.add_argument('--ranks', type=int, default=10)
    parser.add_argument('--length_bucket_num', type=int, default=20)
    parser.add_argument('--max_prompt_length', type=int, default=20)
    parser.add_argument('--max_output_length', type=int, default=100)

    parser.add_argument('--token_decoding_speed', type=float, default=0.05)
    parser.add_argument('--prefill_speed', type=float, default=0.05)
    parser.add_argument('--cache_loading_speed', type=float, default=0.01, help='cache loading speed per token')
    parser.add_argument('--cache_saving_speed', type=float, default=0.01, help='cache saving speed per token')
    args = parser.parse_args()

    def initialize_user_request(args):
        user_request_priority = [random.choice(list(range(args.ranks))) for _ in range(args.user_request_num)]
        user_request_prompt_length = [random.choice(list(range(1, args.max_prompt_length))) for _ in range(args.user_request_num)]
        user_request_output_length = [random.choice(list(range(1, args.max_output_length))) for user_request in user_request_prompt_length]
        return user_request_priority, user_request_prompt_length, user_request_output_length
    
    random.seed(args.seed)
    np.random.seed(args.seed)

    config.user_request_priority, config.user_request_prompt_length, config.user_request_output_length = initialize_user_request(args)

    queue = PriorityQueue(args)
    node = Request(args, 0, 3, 2)
    queue.insert_node(node)
    node = Request(args, 1, 3, 3)
    queue.insert_node(node)
    node = Request(args, 2, 2, 4)
    queue.insert_node(node)
    node = Request(args, 3, 4, 5)
    queue.insert_node(node)
    node = Request(args, 4, 5, 6)
    queue.insert_node(node)
    node = Request(args, 5, 6, 7)
    queue.insert_node(node)

    queue.add_dependency(0, 1)
    queue.add_dependency(2, 0)
    queue.add_dependency(3, 4)
    queue.add_dependency(3, 5)
    queue.add_dependency(3, 2)

    # queue.compute_first_node()
    # queue.fetch_next_node().print_out_features()

    # queue.build_order(list(range(10)))
    # queue.remove_node(5)

    # for user_request_id in range(10, 20):
    #     queue.insert_node(user_request_id)

    # print(queue.get_highest_priority_request_id())

    # queue.remove_node(2)

    # print(queue.get_highest_priority_request_id())

    queue.visualize(queue.root)