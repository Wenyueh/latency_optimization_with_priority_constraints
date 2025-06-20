import argparse, time, random, config, json
import numpy as np
from copy import deepcopy
from queue_management import Request, PriorityQueue
from utils import bcolors, cancel, round_2, compute_average_waiting_time
from compute_arrival_time_utils import *
from semantic_predictor import (
    oracle_priority_predictor, 
    oracle_output_length_bucket_predictor, 
    simulated_priority_predictor, 
    simulated_output_length_bucket_predictor,
    compute_output_time
)
from memory_management import MaxHeap_Memory_Class, compute_GPU_KV_storage_size
import asyncio
import math

REQUEST_BLOCKS = 1
TOTAL_BLOCKS = 1000


def initialize_user_request(args):
    user_request_priority = [random.choice(list(range(args.ranks))) for _ in range(args.user_request_num)]
    user_request_prompt_length = [random.choice(list(range(1, args.max_prompt_length))) for _ in range(args.user_request_num)]
    user_request_output_length = [random.choice(list(range(1, args.max_output_length))) for user_request in user_request_prompt_length]
    return user_request_priority, user_request_prompt_length, user_request_output_length

# whether we preempt user_request_1 by user_request_2
# if user_request_2 cannot preempt any running requests, then no triggering
# if user_request_2 can preempt the highest priority request, then preempt from the lowest
# if user_request_2 can preempt any non-highest priority request, we need to check whether it's prefilling and any higher priority request is decoding

async def simulator(args, user_requests):
    user_request_waiting_time_from_predictor, user_request_arrival_time_from_predictor, time_arrived_requests = compute_arrival_time(args, user_requests)     

    print('requests arrive at specific timestamp:', time_arrived_requests)

    oracle_predicted_priority = oracle_priority_predictor(args, list(range(args.user_request_num)))
    oracle_predicted_output_bucket = oracle_output_length_bucket_predictor(args, list(range(args.user_request_num)))
    simulated_predicted_priority = simulated_priority_predictor(args, oracle_predicted_priority)
    simulated_output_length_bucket = simulated_output_length_bucket_predictor(args, oracle_predicted_output_bucket)
    # Create the priority queue for user request queue management
    full_queue = PriorityQueue(args)
    config.record_requests = {}

    config.ongoing_requests = []
    task_building_queue = asyncio.create_task(simulate_incoming_requests(args, full_queue, time_arrived_requests, simulated_predicted_priority, simulated_output_length_bucket))
    task_GPU_execution = asyncio.create_task(GPU_execute(args, args.MaxHeap_Memory, full_queue))

    await task_building_queue
    await task_GPU_execution
    
    # save_info, file_name = compute_average_waiting_time(args, config.record_requests, user_request_waiting_time_from_predictor)
    
    # return save_info, file_name

async def simulate_incoming_requests(args, full_queue, time_arrived_requests, simulated_predicted_priority, simulated_output_length_bucket):
    for arrival_time_id, arrived_user_request_ids in enumerate(list(time_arrived_requests.values())):
        gap_time = list(time_arrived_requests.keys())[arrival_time_id] - list(time_arrived_requests.keys())[arrival_time_id-1] if arrival_time_id != 0 else list(time_arrived_requests.keys())[arrival_time_id]
        # add incoming requests to the queue
        # and get the next node to process --> max operation
        for user_request_id in arrived_user_request_ids:
            user_request = Request(args, user_request_id, simulated_predicted_priority[user_request_id], simulated_output_length_bucket[user_request_id])
            full_queue.add_unsorted_node(user_request)

        asyncio.create_task(full_queue.incremental_update(), name=f'incremental_update_task_{arrival_time_id}')
        await asyncio.sleep(gap_time)

        full_queue.visualize()

async def GPU_execute(args, MaxHeap_Memory, full_queue):
    def one_execute_iteration(args, ongoing_requests):
        prefilling_requests = []
        # check whether there's prefilling requests in the batch
        for user_request in ongoing_requests:
            if user_request.predicted_remaining_computation_time[0] > 0:
                prefilling_requests.append(user_request)

        MaxHeap_Memory.update_ongoing_requests(ongoing_requests)

        # allocate memory for each request (token-wise)
        preempted_requests_memory_killed = []
        deallocated_requests = []
        if prefilling_requests != []:
            # allocate memory for prefilling [user_request.prompt_length tokens]
            for user_request in prefilling_requests:
                res = MaxHeap_Memory.allocate_memory_for(user_request, user_request.prompt_length, full_queue)
                preempted_requests_memory_killed.extend(res['preempted_requests'])
                deallocated_requests.extend(res['deallocated_requests'])
        else:
            # allocate memory for decoding [1 token]
            for user_request in ongoing_requests:
                res = MaxHeap_Memory.allocate_memory_for(user_request, 1, full_queue)
                preempted_requests_memory_killed.extend(res['preempted_requests'])
                deallocated_requests.extend(res['deallocated_requests'])

        # remove preempted requests from prefilling_requests and ongoing_requests
        prefilling_requests = [user_request for user_request in prefilling_requests if user_request.user_request_id not in [r.user_request_id for r in preempted_requests_memory_killed]]
        ongoing_requests = [user_request for user_request in ongoing_requests if user_request.user_request_id not in [r.user_request_id for r in preempted_requests_memory_killed]]

        # update and push deallocated_requests back to full_queue
        for deallocated_request in deallocated_requests:
            full_queue.two_dim_priority_queue.push(deallocated_request, deallocated_request.predicted_priority, deallocated_request.predicted_remaining_computation_time[-1] + deallocated_request.prefill_cache_loading_time + deallocated_request.decoding_cache_loading_time)

        if prefilling_requests != []:
            # the whole prefilling is one iteration
            iteration_time = max([user_request.remaining_computation_time[0] + user_request.prefill_cache_loading_time for user_request in prefilling_requests])
            for user_request in prefilling_requests:
                user_request.update_normal_prefilling_iteration()
        else:
            # one decoded token is one iteration
            decoded_cache_length = max(user_request.CPU_decoding_loaded_cache) if user_request.CPU_decoding_loaded_cache != [] else max(0, MaxHeap_Memory.request_id2tokens[user_request.user_request_id]-1 - user_request.prompt_length)
            iteration_time = max([compute_output_time(args, user_request.prompt_length + decoded_cache_length, 1)+user_request.decoding_cache_loading_time for user_request in ongoing_requests])
            for user_request in ongoing_requests:
                user_request.update_normal_decoding_iteration(MaxHeap_Memory)

        # update remaining computation time for all requests on MinHeap
        for user_request in deallocated_requests:
            # remove the lowest_priority_request from full_queue minheap
            full_queue.two_dim_priority_queue.delete(user_request)
            # update remaining time for the lowest_priority_request and reinsert it to the maxheap
            user_request.swap_or_delete_update(remaining_tokens_on_GPU=MaxHeap_Memory.request_id2tokens[user_request.user_request_id])
            remaining_time = user_request.predicted_remaining_computation_time[-1] + user_request.prefill_cache_loading_time + user_request.decoding_cache_loading_time
            full_queue.two_dim_priority_queue.push(user_request, user_request.predicted_priority, remaining_time)
        
        # update MaxHeap_Memory with updated remaining computation time for ongoing_requests
        MaxHeap_Memory.reconstruct_heap(ongoing_requests)

        ongoing_requests = [user_request for user_request in ongoing_requests if user_request.remaining_computation_time[-1] > 0]
        
        return iteration_time, ongoing_requests

    def call_scheduler(args, full_queue, ongoing_requests):
        next_node = full_queue.fetch_next_node()
        ######### find whether we should do prefilling or decoding in preemption #########
        ##################################################################################
        # compare next_node & current highest_priority node
        if next_node.predicted_priority < config.ongoing_requests[0].predicted_priority or (next_node.predicted_priority == config.ongoing_requests[0].predicted_priority and next_node.predicted_remaining_computation_time[-1] < config.ongoing_requests[0].predicted_remaining_computation_time[-1]):
            node_to_align = next_node
        else:
            node_to_align = config.ongoing_requests[0]
            full_queue.two_dim_priority_queue.push(next_node, next_node.predicted_priority, next_node.predicted_remaining_computation_time[-1] + next_node.prefill_cache_loading_time + next_node.decoding_cache_loading_time)

        require_decode = False
        max_prefilling_time = float('inf')
        if node_to_align.predicted_remaining_computation_time[0] <= 0:
            require_decode = True 
        else:
            max_prefilling_time = node_to_align.predicted_remaining_computation_time[0]
        
        ######################################################
        
        # fetch the highest ordered request in the newly incoming batch of requests
        # return a sorted list
        next_batch = full_queue.fetch_next_k_nodes(args.batch_size, require_decode=require_decode, max_prefilling_time=max_prefilling_time)
        # for each pair of user requests
        # user_request_1 is the ongoing request
        # user_request_2 is the next request that can preempt user_request_1

        def merge(list1, list2):
            if len(list1) == 0:
                return list2
            if len(list2) == 0:
                return list1
            # merge two list of user requests based on (predicted_priority, predicted_remaining_computation_time)
            pq = []
            i, j = 0, 0
            while i < len(list1) and j < len(list2):
                if list1[i].predicted_priority < list2[j].predicted_priority or (list1[i].predicted_priority == list2[j].predicted_priority and list1[i].predicted_remaining_computation_time[-1] < list2[j].predicted_remaining_computation_time[-1]):
                    pq.append(list1[i])
                    i += 1
                else:
                    pq.append(list2[j])
                    j += 1
            if i < len(list1):
                pq += list1[i:]
            if j < len(list2):
                pq += list2[j:]
            return pq

        merged_queue = merge(ongoing_requests, next_batch)
        top_batch_size_requests = merged_queue[:args.batch_size]
        push_back_requests = [merged_queue[i] for i in range(len(merged_queue)) if merged_queue[i].user_request_id not in [r.user_request_id for r in top_batch_size_requests]]

        # put back the push back requests
        for user_request in push_back_requests:
            full_queue.two_dim_priority_queue.push(user_request, user_request.predicted_priority, user_request.predicted_remaining_computation_time[-1] + user_request.prefill_cache_loading_time + user_request.decoding_cache_loading_time)
        
        return top_batch_size_requests
    
    # when there's super long gap between next incoming batch of requests
    # we wait maximum 20 seconds
    # if no requests coming, we stop the simulation
    async def detect_requests_coming(full_queue):
        max_wait_in_simulation = 2000
        n = 0
        while len(full_queue.two_dim_priority_queue) == 0 and len(config.ongoing_requests) == 0 and len(full_queue.unsorted_nodes) == 0:
            await asyncio.sleep(0.01)
            n += 1
            if n > max_wait_in_simulation:
                return False
        return True

    has_requests = True
    while has_requests:
        if len(config.ongoing_requests) == 0:
            config.ongoing_requests = full_queue.fetch_next_k_nodes(args.batch_size, require_decode=False, max_prefilling_time=float('inf'))
        while (len(full_queue.two_dim_priority_queue) + len(full_queue.unsorted_nodes) > 0) or len(config.ongoing_requests) != 0:
            # run one iteration
            iteration_time, config.ongoing_requests = one_execute_iteration(args, config.ongoing_requests)
            await asyncio.sleep(iteration_time)
            # update remaining computation time
            for request in config.ongoing_requests:
                # just for record to compute final result
                config.record_requests[request.user_request_id] = request
            # update waiting time for all requests
            full_queue.update_waiting_time(iteration_time)

            # delete completed requests' memory block and remove them from ongoing_requests
            completed_requests = [r for r in config.ongoing_requests if r.remaining_computation_time[-1] <= 0]
            for user_request in completed_requests:
                MaxHeap_Memory.deallocate_memory_for(user_request)
            config.ongoing_requests = [r for r in config.ongoing_requests if r not in completed_requests]

            # some visualization
            MaxHeap_Memory.visualize()
            full_queue.visualize()

            # check scheduler
            try:
                config.ongoing_requests = call_scheduler(args, full_queue, config.ongoing_requests)
            except Exception as e:
                print("Warning from scheduler: " + str(e))
        
        has_requests = await detect_requests_coming(full_queue)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gap_num', type=int, default=100)
    parser.add_argument('--GPU_memory', type=float, default=40)
    parser.add_argument('--GPU_memory_utilization', type=float, default=0.8)
    parser.add_argument('--activation_memory_percentage', type=float, default=0.05)
    parser.add_argument('--block_size', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=16)

    parser.add_argument('--priority_error_distance', type=float, default=0)
    parser.add_argument('--priority_error_rate', type=float, default=0)
    parser.add_argument('--output_length_error_distance', type=float, default=0)
    parser.add_argument('--output_length_error_rate', type=float, default=0)
    parser.add_argument('--processing_mode', type=str, default='immediate_processing', help='immediate_processing or fully_delayed_processing')

    parser.add_argument('--user_request_gap', type=float, default=1)
    parser.add_argument('--max_concurrent_user_requests', type=int, default=5)
    parser.add_argument('--priority_predictor_batching_size', type=int, default=5)
    parser.add_argument('--priority_predictor_latency', type=float, default=0.003)
    parser.add_argument('--length_predictor_batching_size', type=int, default=5)
    parser.add_argument('--length_predictor_latency', type=float, default=0.003)

    parser.add_argument('--ranks', type=int, default=5)
    parser.add_argument('--length_bucket_num', type=int, default=5)
    parser.add_argument('--max_prompt_length', type=int, default=500)
    parser.add_argument('--max_output_length', type=int, default=500)

    parser.add_argument('--setting', type=str, default='A100_7B_', help='A100_4B, A100_7B, A5000_4B, A5000_7B')
    args = parser.parse_args()

    if args.setting == 'A5000_4B_':
        args.token_decoding_speed_coefficient1 = 2.86938135e-06
        args.token_decoding_speed_coefficient2 = 3.00834536e-02
        args.prefill_speed_coefficient1 = 1.58625005e-09
        args.prefill_speed_coefficient2 = 1.30542054e-04
        args.cache_loading_speed = 0.0003
        args.cache_saving_speed = 0.0003
    elif args.setting == 'A100_4B_':
        args.token_decoding_speed_coefficient1 = 5.91298857e-09
        args.token_decoding_speed_coefficient2 = 1.195828e-02
        args.prefill_speed_coefficient1 = 1.46584975e-09
        args.prefill_speed_coefficient2 = 1.0515576e-04
        args.cache_loading_speed = 0.0001
        args.cache_saving_speed = 0.0001
    elif args.setting == 'A100_7B_':
        args.token_decoding_speed_coefficient1 = 1.34911080e-08
        args.token_decoding_speed_coefficient2 = 1.330198e-02
        args.prefill_speed_coefficient1 = 5.13462572e-07
        args.prefill_speed_coefficient2 = 1.48057167e-04
        args.cache_loading_speed = 0.0001
        args.cache_saving_speed = 0.0001
    elif args.setting == 'A5000_7B_':
        args.token_decoding_speed_coefficient1 = 2.11725653e-06
        args.token_decoding_speed_coefficient2 = 2.72656264e-02
        args.prefill_speed_coefficient1 = 1.85905340e-09
        args.prefill_speed_coefficient2 = 2.17527368e-04
        args.cache_loading_speed = 0.0003
        args.cache_saving_speed = 0.0003

    random.seed(args.seed)
    np.random.seed(args.seed)
    
    args.GPU_KV_cache, args.KV_block_number = compute_GPU_KV_storage_size(args)
    args.MaxHeap_Memory = MaxHeap_Memory_Class(args)
    
    # for each time interval, we have a random number of user requests coming
    num_of_user_requests_coming = [random.randint(0, args.max_concurrent_user_requests) for _ in range(args.gap_num)]

    # total number of user requests coming
    args.user_request_num = sum(num_of_user_requests_coming)
    # initialize the user requests
    user_requests = []
    total_number = 0
    for number_of_user_requests in num_of_user_requests_coming:
        user_requests.append(list(range(total_number, total_number+number_of_user_requests)))
        total_number += number_of_user_requests

    config.user_request_priority, config.user_request_prompt_length, config.user_request_output_length = initialize_user_request(args)
    
    #save_info, file_name = 
    asyncio.run(simulator(args, user_requests))
    # # compute total time
    # total_waiting_time = sum([v['waiting_time'] for k,v in save_info.items() if k != 'average_waiting_time_with_priority'])
    # save_info['total_waiting_time'] = total_waiting_time
    # file_name = args.setting + file_name
    # print(file_name)
    # # save the information
    # with open(file_name, 'w') as f:
    #     json.dump(save_info, f, indent=4)
