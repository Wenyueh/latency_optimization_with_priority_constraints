import argparse
import numpy as np
import random
import config
from copy import deepcopy
from queue_management import Request, PriorityQueue
from utils import bcolors, cancel, round_2, compute_average_waiting_time
from semantic_predictor import (
    oracle_priority_predictor, 
    oracle_output_length_bucket_predictor, 
    update_cache_loading_or_recomputation_time_and_extra_saving_time, 
)
import asyncio
import math

def initialize_user_request(args):
    user_request_priority = [random.choice(list(range(args.ranks))) for _ in range(args.user_request_num)]
    user_request_prompt_length = [random.choice(list(range(1, args.max_prompt_length))) for _ in range(args.user_request_num)]
    user_request_output_length = [random.choice(list(range(1, args.max_output_length))) for user_request in user_request_prompt_length]
    return user_request_priority, user_request_prompt_length, user_request_output_length

# whether we preempt user_request_1 by user_request_2
def trigger_preemption(args, user_request_1, user_request_2):
    if user_request_1.predicted_priority > user_request_2.predicted_priority:
        return True 
    elif user_request_1.predicted_priority < user_request_2.predicted_priority:
        return False
    elif user_request_1.predicted_priority == user_request_2.predicted_priority:
        # copy the user_request_1 object
        user_request_1_copy = deepcopy(user_request_1)
        # add loading_cache/recomputation time to total remaining time of the preempted request
        user_request_1_copy_save_time = update_cache_loading_or_recomputation_time_and_extra_saving_time(args, user_request_1_copy)

        # if preempting: first user_request_2, then user_request_1
        preempting_time_user_request_2 = user_request_1_copy_save_time + user_request_2.prefill_cache_loading_time + user_request_2.decoding_cache_loading_time + user_request_2.predicted_remaining_computation_time[-1]
        preempting_time_user_request_1_copy = user_request_1_copy.prefill_cache_loading_time + user_request_1_copy.decoding_cache_loading_time + user_request_1_copy.predicted_remaining_computation_time[-1]
        preempting_total_waiting_time = 2*preempting_time_user_request_2 + preempting_time_user_request_1_copy
        
        # if not preempting: first user_request_1, then user_request_2
        left_prefill_loading_time = (user_request_1.prefill_cache_proportion-user_request_1.prefill_cache_loaded_proportion) * user_request_1.prefill_cache_loading_time
        left_decoding_loading_time = (user_request_1.decoding_cache_length-user_request_1.decoding_cache_loaded_length) * args.cache_loading_speed
        no_preempting_time_user_request_1 = left_prefill_loading_time + left_decoding_loading_time + user_request_1.predicted_remaining_computation_time[-1]
        no_preempting_time_user_request_2 = user_request_2.prefill_cache_loading_time + user_request_2.decoding_cache_loading_time + user_request_2.predicted_remaining_computation_time[-1]
        no_preempting_total_waiting_time = 2*no_preempting_time_user_request_1 + no_preempting_time_user_request_2

        if preempting_total_waiting_time < no_preempting_total_waiting_time:
            return True
        else:
            return False

def compute_arrival_time(args, user_requests):
    waiting_time = {}
    arrival_time = {}
    user_requests_order = {tuple(one_batch): user_requests.index(one_batch) for one_batch in user_requests}
    requests_coming_order = {}
    for k,v in user_requests_order.items():
        for request_id in k:
            requests_coming_order[request_id] = v
    all_requests = [i for one_batch_user_request_ids in user_requests for i in one_batch_user_request_ids ]
    length_computing_batch = [all_requests[i:i+args.length_predictor_batching_size] for i in range(0, len(all_requests), args.length_predictor_batching_size)]
    priority_computing_batch = [all_requests[i:i+args.priority_predictor_batching_size] for i in range(0, len(all_requests), args.priority_predictor_batching_size)]

    length_arrival_time = {}
    priority_arrival_time = {}
    length_waiting_time = {}
    priority_waiting_time = {}
    for request_id in all_requests:
        length_computing_batch_number = request_id//args.length_predictor_batching_size
        max_coming_order = max([requests_coming_order[request_id] for request_id in length_computing_batch[length_computing_batch_number]])
        length_arrival_time[request_id] = max_coming_order*args.user_request_gap + args.length_predictor_latency
        length_waiting_time[request_id] = (max_coming_order - requests_coming_order[request_id])*args.user_request_gap + args.length_predictor_latency

        priority_computing_batch_number = request_id//args.priority_predictor_batching_size
        max_coming_order = max([requests_coming_order[request_id] for request_id in priority_computing_batch[priority_computing_batch_number]])
        priority_arrival_time[request_id] = max_coming_order*args.user_request_gap + args.priority_predictor_latency
        priority_waiting_time[request_id] = (max_coming_order - requests_coming_order[request_id])*args.user_request_gap + args.priority_predictor_latency

    for request_id in all_requests:
        arrival_time[request_id] = max(length_arrival_time[request_id], priority_arrival_time[request_id])
        waiting_time[request_id] = max(length_waiting_time[request_id], priority_waiting_time[request_id])

    time_arrived_requests = {}
    for user_request_id, arrive_time in arrival_time.items():
        if arrive_time in time_arrived_requests:
            time_arrived_requests[arrive_time].append(user_request_id)
        else:
            time_arrived_requests[arrive_time] = [user_request_id]

    return waiting_time, arrival_time, time_arrived_requests

async def GPU_execute(arrival_time_id, arrival_intervals, full_queue, ongoing_request, record_requests):
    # if it is not the last batch
    if arrival_time_id < len(arrival_intervals) - 1:
        # compute the running time for this interval, before the next batch of requests arrive
        running_time_for_this_interval = round_2(arrival_intervals[arrival_time_id+1] - arrival_intervals[arrival_time_id])
        # total running time required = load cache first and then compute the rest

        # remaining loading time of cache
        prefill_cache_loading_time_remained = round_2(ongoing_request.prompt_length * ongoing_request.args.cache_loading_speed * (ongoing_request.prefill_cache_proportion-ongoing_request.prefill_cache_loaded_proportion))
        decoding_cache_loading_time_remained = round_2(ongoing_request.args.cache_loading_speed * (ongoing_request.decoding_cache_length-ongoing_request.decoding_cache_loaded_length))

        full_time_required = ongoing_request.previous_save_time + prefill_cache_loading_time_remained + decoding_cache_loading_time_remained + ongoing_request.remaining_computation_time[-1]
        
        # if the running time for this interval is longer than the current request
        # we just keep putting the next highest priority request to the GPU while at the same time computing the next highest priority request
        # until the running time for this interval is used up
        # otherwise, we just run the current request and compute the next highest priority request
        if running_time_for_this_interval > full_time_required:
            while running_time_for_this_interval > full_time_required and len(full_queue.two_dim_priority_queue.heap) !=0:
                await asyncio.sleep(full_time_required)

                # put the currently highest priority user request to the GPU
                # update both actual time and predicted time
                ongoing_request.update_computation_time_normal_run(full_time_required)
                full_queue.update_waiting_time(full_time_required)
                ongoing_request.update_waiting_time(full_time_required)

                # update log
                record_requests[ongoing_request.user_request_id] = ongoing_request

                ongoing_request.print_out_features()
                full_queue.visualize()
                print('-'*50)
                print('-'*50)
                
                # running time for the next request
                running_time_for_this_interval -= full_time_required

                ##### write one function to compute the next highest priority request asynchronously #####
                # put the currently highest priority user request to the GPU
                ongoing_request = full_queue.fetch_next_node()
                ##### write one function to compute the next highest priority request asynchronously #####

                # need to load cache first and then compute the rest
                prefill_cache_loading_time_remained = round_2(ongoing_request.prompt_length * ongoing_request.args.cache_loading_speed * (ongoing_request.prefill_cache_proportion-ongoing_request.prefill_cache_loaded_proportion))
                decoding_cache_loading_time_remained = round_2(ongoing_request.args.cache_loading_speed * (ongoing_request.decoding_cache_length-ongoing_request.decoding_cache_loaded_length))
                full_time_required = ongoing_request.previous_save_time + prefill_cache_loading_time_remained + decoding_cache_loading_time_remained + ongoing_request.remaining_computation_time[-1]

                # if we run out of the running time for this interval
                if running_time_for_this_interval <= full_time_required:
                    # asynchronously compute the next node
                    await asyncio.sleep(running_time_for_this_interval)
                    # put the currently highest priority user request to the GPU
                    ongoing_request.update_computation_time_normal_run(running_time_for_this_interval)
                    full_queue.update_waiting_time(running_time_for_this_interval)
                    ongoing_request.update_waiting_time(running_time_for_this_interval)
                    record_requests[ongoing_request.user_request_id] = ongoing_request

                    ongoing_request.print_out_features()
                    full_queue.visualize()
                    print('-'*50)
                    print('-'*50)

                elif (len(full_queue.two_dim_priority_queue.heap) + len(full_queue.unsorted_nodes)) == 0:
                    # if there is no more request in the queue, and the ongoing request is also completed
                    await asyncio.sleep(running_time_for_this_interval)
                    # put the currently highest priority user request to the GPU
                    ongoing_request.update_computation_time_normal_run(ongoing_request.remaining_computation_time[-1])
                    ongoing_request.update_waiting_time(ongoing_request.remaining_computation_time[-1])
                    record_requests[ongoing_request.user_request_id] = ongoing_request

                    ongoing_request.print_out_features()
                    ongoing_request = None
                    full_queue.visualize()
                    print('-'*50)
                    print('-'*50)
        else:
            # run the currently highest priority user request on the GPU
            ongoing_request.update_computation_time_normal_run(running_time_for_this_interval)
            full_queue.update_waiting_time(running_time_for_this_interval)
            ongoing_request.update_waiting_time(running_time_for_this_interval)
            record_requests[ongoing_request.user_request_id] = ongoing_request
            await asyncio.sleep(running_time_for_this_interval)

            ongoing_request.print_out_features()
            print('-'*50)
            print('-'*50)
    
    else:
        print(bcolors.WARNING + 'Last batch of requests' + bcolors.ENDC)
        # after we have put all requests in the queue, we just process 1 by 1
        while (len(full_queue.two_dim_priority_queue.heap) + len(full_queue.unsorted_nodes)) != 0:
            # remaining loading time of cache
            prefill_cache_loading_time_remained = round_2(ongoing_request.prompt_length * ongoing_request.args.cache_loading_speed * (ongoing_request.prefill_cache_proportion-ongoing_request.prefill_cache_loaded_proportion))
            decoding_cache_loading_time_remained = round_2(ongoing_request.args.cache_loading_speed * (ongoing_request.decoding_cache_length-ongoing_request.decoding_cache_loaded_length))
            running_time_for_this_interval = ongoing_request.previous_save_time + prefill_cache_loading_time_remained + decoding_cache_loading_time_remained + ongoing_request.remaining_computation_time[-1]

            # put the currently highest priority user request to the GPU
            ongoing_request.update_computation_time_normal_run(running_time_for_this_interval)
            full_queue.update_waiting_time(running_time_for_this_interval)
            ongoing_request.update_waiting_time(running_time_for_this_interval)
            record_requests[ongoing_request.user_request_id] = ongoing_request

            print('running_time_for_this_interval')
            print(running_time_for_this_interval)

            await asyncio.sleep(running_time_for_this_interval)

            ongoing_request.print_out_features()
            full_queue.visualize()
            print('-'*50)
            print('-'*50)

            ongoing_request = full_queue.fetch_next_node()

            if (len(full_queue.two_dim_priority_queue.heap) + len(full_queue.unsorted_nodes)) == 0:
                # remaining loading time of cache
                prefill_cache_loading_time_remained = round_2(ongoing_request.prompt_length * ongoing_request.args.cache_loading_speed * (ongoing_request.prefill_cache_proportion-ongoing_request.prefill_cache_loaded_proportion))
                decoding_cache_loading_time_remained = round_2(ongoing_request.args.cache_loading_speed * (ongoing_request.decoding_cache_length-ongoing_request.decoding_cache_loaded_length))
                running_time_for_this_interval = ongoing_request.previous_save_time + prefill_cache_loading_time_remained + decoding_cache_loading_time_remained + ongoing_request.remaining_computation_time[-1]
            
                ongoing_request.update_computation_time_normal_run(running_time_for_this_interval)
                ongoing_request.update_waiting_time(running_time_for_this_interval)
                record_requests[ongoing_request.user_request_id] = ongoing_request
                await asyncio.sleep(running_time_for_this_interval)

                ongoing_request.print_out_features()
                full_queue.visualize()
                print('-'*50)
                print('-'*50)

    full_queue.is_GPU_job_completed = True 

async def simulator(args, user_requests):
    user_request_waiting_time_from_predictor, user_request_arrival_time_from_predictor, time_arrived_requests = compute_arrival_time(args, user_requests)     
    
    print('requests arrive at specific timestamp:', time_arrived_requests)

    arrival_intervals = list(time_arrived_requests.keys())
    oracle_predicted_priority = oracle_priority_predictor(list(range(args.user_request_num)))
    oracle_predicted_output_bucket = oracle_output_length_bucket_predictor(args, list(range(args.user_request_num)))

    # Create the priority queue for user request queue management
    full_queue = PriorityQueue(args)
    record_requests = {}

    ongoing_request = None
    for arrival_time_id, arrived_user_request_ids in enumerate(list(time_arrived_requests.values())):
        # add incoming requests to the queue
        # and get the next node to process --> max operation
        for user_request_id in arrived_user_request_ids:
            user_request = Request(args, user_request_id, oracle_predicted_priority[user_request_id], oracle_predicted_output_bucket[user_request_id])
            full_queue.add_unsorted_node(user_request)

        asyncio.create_task(full_queue.incremental_update(), name=f'incremental_update_task_{arrival_time_id}')

        if ongoing_request is None:
            # fetch the next node
            ongoing_request = full_queue.fetch_next_node()
        else:
            # fetch the highest ordered request in the newly incoming batch of requests
            next_request_in_batch = full_queue.fetch_next_node()

            # preempt the ongoing request by the new request if necessary
            if trigger_preemption(args, ongoing_request, next_request_in_batch):
                print(bcolors.WARNING + f"Preempting {ongoing_request.user_request_id} by {next_request_in_batch.user_request_id}" + bcolors.ENDC)
                
                print('ongoing_request')
                ongoing_request.print_out_features()
                print('next_request_in_batch')
                next_request_in_batch.print_out_features()

                # add loading_cache/recomputation time to total remaining time of the preempted request
                ongoing_request_cache_save_time = update_cache_loading_or_recomputation_time_and_extra_saving_time(args, ongoing_request)

                # put the preempted request back to the queue
                full_queue.two_dim_priority_queue.push(ongoing_request, ongoing_request.predicted_priority, ongoing_request.predicted_remaining_computation_time[-1] + ongoing_request.prefill_cache_loading_time + ongoing_request.decoding_cache_loading_time)

                # next time, we need to reload the cache
                ongoing_request.prefill_cache_loaded_proportion = 0
                ongoing_request.decoding_cache_loaded_length = 0

                # the preempting request is now the ongoing request
                ongoing_request = next_request_in_batch
                # we also consider the situation where the previous saving time is not 0
                # meaning that the request before the preempted request has not finished saving the cache
                # in which case ongoing_request_cache_save_time = 0, but we need to add the previous saving time to the current request
                ongoing_request.previous_save_time = ongoing_request_cache_save_time + ongoing_request.previous_save_time
            else:
                print(bcolors.WARNING + "No preemption" + bcolors.ENDC)
                # add next_request_in_batch in queue right after the root
                full_queue.two_dim_priority_queue.push(next_request_in_batch, ongoing_request.predicted_priority, next_request_in_batch.predicted_remaining_computation_time[-1] + next_request_in_batch.prefill_cache_loading_time + ongoing_request.decoding_cache_loading_time)
        
        if arrival_time_id < len(arrival_intervals) - 1:
            asyncio.create_task(GPU_execute(arrival_time_id, arrival_intervals, full_queue, ongoing_request, record_requests), name=f'GPU_execute_task_{ongoing_request.user_request_id}')
            running_time = round_2(arrival_intervals[arrival_time_id+1] - arrival_intervals[arrival_time_id])
            await asyncio.sleep(running_time)
            full_queue.visualize()
        else:
            await GPU_execute(arrival_time_id, arrival_intervals, full_queue, ongoing_request, record_requests)

    waiting_time_with_priority = compute_average_waiting_time(record_requests)
    print(waiting_time_with_priority)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--user_request_gap', type=float, default=0.5)
    parser.add_argument('--priority_predictor_batching_size', type=int, default=5)
    parser.add_argument('--priority_predictor_latency', type=float, default=0.7)
    parser.add_argument('--length_predictor_batching_size', type=int, default=2)
    parser.add_argument('--length_predictor_latency', type=float, default=0.7)
    parser.add_argument('--user_request_waiting', type=int, default=5)
    parser.add_argument('--simulation_scale', type=float, default=0.01)

    parser.add_argument('--gap_num', type=int, default=10)
    parser.add_argument('--ranks', type=int, default=10)
    parser.add_argument('--length_bucket_num', type=int, default=20)
    parser.add_argument('--max_prompt_length', type=int, default=20)
    parser.add_argument('--max_output_length', type=int, default=100)

    parser.add_argument('--token_decoding_speed', type=float, default=0.005)
    parser.add_argument('--prefill_speed', type=float, default=0.01)
    parser.add_argument('--cache_loading_speed', type=float, default=0.015, help='cache loading speed per token')
    parser.add_argument('--cache_saving_speed', type=float, default=0.015, help='cache saving speed per token')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # for each time interval, we have a random number of user requests coming
    num_of_user_requests_coming = [random.randint(0, 5) for _ in range(args.gap_num)]
    # total number of user requests coming
    args.user_request_num = sum(num_of_user_requests_coming)
    # initialize the user requests
    user_requests = []
    total_number = 0
    for number_of_user_requests in num_of_user_requests_coming:
        user_requests.append(list(range(total_number, total_number+number_of_user_requests)))
        total_number += number_of_user_requests

    config.user_request_priority, config.user_request_prompt_length, config.user_request_output_length = initialize_user_request(args)

    asyncio.run(simulator(args, user_requests))
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
