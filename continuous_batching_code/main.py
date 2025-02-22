import argparse, time, random, config, json
import numpy as np
from copy import deepcopy
from queue_management import Request, PriorityQueue
from utils import bcolors, cancel, round_2, compute_average_waiting_time, merge, topk
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
    asyncio.create_task(simulate_incoming_requests(args, full_queue, simulated_predicted_priority, simulated_output_length_bucket))
    asyncio.create_task(GPU_execute(full_queue))
    
    save_info, file_name = compute_average_waiting_time(args, config.record_requests, user_request_waiting_time_from_predictor)
    
    return save_info, file_name

async def simulate_incoming_requests(args, full_queue, simulated_predicted_priority, simulated_output_length_bucket):
    for arrival_time_id, arrived_user_request_ids in enumerate(list(time_arrived_requests.values())):
        gap_time = list(time_arrived_requests.keys())[arrival_time_id] - list(time_arrived_requests.keys())[arrival_time_id-1] if arrival_time_id != 0 else list(time_arrived_requests.keys())[arrival_time_id]
        # add incoming requests to the queue
        # and get the next node to process --> max operation
        for user_request_id in arrived_user_request_ids:
            user_request = Request(args, user_request_id, simulated_predicted_priority[user_request_id], simulated_output_length_bucket[user_request_id])
            full_queue.add_unsorted_node(user_request)

        asyncio.create_task(full_queue.incremental_update(), name=f'incremental_update_task_{arrival_time_id}')

        await asyncio.sleep(gap_time/10)

async def GPU_execute(full_queue):
    def one_execute_iteration(ongoing_requests):
        prefilling_requests = []
        # check whether there's prefilling requests in the batch
        for user_request in ongoing_requests:
            if user_request.predicted_remaining_computation_time[0] > 0:
                prefilling_requests.append(request)
        if prefilling_requests != []:
            # the whole prefilling is one iteration
            iteration_time = max([user_request.remaining_computation_time[0] for user_request in prefilling_requests])
            for user_request in prefilling_requests:
                user_request.update_computation_time_normal_run(user_request.remaining_computation_time[0])
            for user_request in ongoing_requests:
                user_request.update_waiting_time(iteration_time)
        else:
            # one decoded token is one iteration
            iteration_time = max([compute_output_time(args, user_request.prompt_length + len(user_request.decoding_cache_position), 1) for user_request in ongoing_requests])
            for user_request in prefilling_requests:
                user_request.update_computation_time_normal_run(compute_output_time(args, user_request.prompt_length + len(user_request.decoding_cache_position), 1))
            for request in ongoing_requests:
                user_request.update_waiting_time(iteration_time)

        ongoing_requests = [user_request for user_request in ongoing_requests if user_request.remaining_computation_time[-1] > 0]
        return iteration_time, ongoing_requests

    def call_scheduler(args, full_queue, ongoing_requests):
        next_node = full_queue.fetch_next_node()
        ######### find whether we should do prefilling or decoding in preemption #########
        ######################################################
        # compare next_node & current highest_priority node
        if next_node.predicted_priority < config.ongoing_requests[0].predicted_priority or (next_node.predicted_priority == config.ongoing_requests[0].predicted_priority and next_node.predicted_remaining_computation_time[-1] < config.ongoing_requests[0].predicted_remaining_computation_time[-1]):
            node_to_align = next_node
        else:
            node_to_align = config.ongoing_requests[0]
        require_decode = False
        max_prefilling_time = float('inf')
        if node_to_align.predicted_remaining_computation_time[0] == 0:
            require_decode = True 
        else:
            max_prefilling_time = node_to_align.predicted_remaining_computation_time[0]
        full_queue.push(next_node, next_node.predicted_priority, next_node.predicted_remaining_computation_time[-1] + next_node.prefill_cache_loading_time + next_node.decoding_cache_loading_time)
        ######################################################
        
        # fetch the highest ordered request in the newly incoming batch of requests
        # return a sorted list
        next_batch = full_queue.fetch_next_k_nodes(args.batch_size, require_decode=require_decode, max_prefilling_time=max_prefilling_time)

        # for each pair of user requests
        # user_request_1 is the ongoing request
        # user_request_2 is the next request that can preempt user_request_1

        def merge(list1, list2):
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
        preempted_requests = [i for i in range(len(ongoing_requests)) if ongoing_requests[i].user_request_id not in [r.user_request_id for r in top_batch_size_requests]]
        push_back_requests = [i for i in range(len(merged_queue)) if merged_queue[i].user_request_id not in [r.user_request_id for r in top_batch_size_requests]]

        #!! request memory block, swap together or delete (deleted requests)
        # allocate before computing the iteration
        # we have max bandwidth size, might need sequential but unlikely
        # if requests memory belongs to ongoing batch, remove it from batch
        # Dujian

        #!! update other requests' remaining computation time due to deletion or swapping
        # remaining time = CPU loading time + recomputation time + original remaining time
        # Wenyue

        return top_batch_size_requests, preempted_requests, push_back_requests
    
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
        while len(full_queue.two_dim_priority_queue) != 0 or len(config.ongoing_requests) != 0 or len(full_queue.unsorted_nodes) != 0:
            # run one iteration
            iteration_time, config.ongoing_requests = one_execute_iteration(config.ongoing_requests)
            # check scheduler
            config.ongoing_requests, preempted_requests, push_back_requests = call_scheduler(args, full_queue, config.ongoing_requests)
            # update remaining computation time
            for request in config.ongoing_requests:
                request.update_computation_time_normal_run(iteration_time)
                # just for record to compute final result
                config.record_requests[request.user_request_id] = request
            # update waiting time for all requests
            full_queue.update_waiting_time(iteration_time)

            # put back the push back requests
            for user_request in push_back_requests:
                full_queue.two_dim_priority_queue.push(user_request, user_request.predicted_priority, user_request.predicted_remaining_computation_time[-1] + user_request.prefill_cache_loading_time + user_request.decoding_cache_loading_time)
            completed_requests = [r for r in config.ongoing_requests if r.remaining_computation_time[-1] == 0]
            #!! delete completed requests' memory block
            # Dujian

            await asyncio.sleep(iteration_time/10)
        
        has_requests = asyncio.run(detect_requests_coming(full_queue))


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
    
    print(user_requests)

    config.user_request_priority, config.user_request_prompt_length, config.user_request_output_length = initialize_user_request(args)
    
    save_info, file_name = asyncio.run(simulator(args, user_requests))
    # compute total time
    total_waiting_time = sum([v['waiting_time'] for k,v in save_info.items() if k != 'average_waiting_time_with_priority'])
    save_info['total_waiting_time'] = total_waiting_time
    file_name = args.setting + file_name
    print(file_name)
    # save the information
    with open(file_name, 'w') as f:
        json.dump(save_info, f, indent=4)

    print('Total waiting time: ', total_waiting_time)
    print(save_info['average_waiting_time_with_priority'])

    # queue = PriorityQueue(args)
    # for user_request_id in range(10):
    #     node = Request(args, user_request_id)
    #     print(node.user_request_id)
    #     print(node.predicted_priority)
    #     print(node.computation_time)
    #     queue.insert_node(user_request_id)

    # queue.build_order(list(range(10)))
    # queue.remove_node(5)

    # for user_request_id in range(10, 20):
    #     queue.insert_node(user_request_id)

    # print(queue.get_highest_priority_request_id())

    # queue.remove_node(2)

    # print(queue.get_highest_priority_request_id())

    # queue.visualize(queue.root)

    # waiting_time, arrival_time = compute_arrival_time(args)
    # print(waiting_time)
    # print(arrival_time)
