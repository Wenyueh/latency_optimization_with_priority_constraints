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

async def GPU_execute(arrival_time_id, arrival_intervals, full_queue, ongoing_request, record_requests):
    config.ongoing_request = ongoing_request
    config.record_requests = record_requests
    # if it is not the last batch
    if arrival_time_id < len(arrival_intervals) - 1:
        # compute the running time for this interval, before the next batch of requests arrive
        running_time_for_this_interval = round_2(arrival_intervals[arrival_time_id+1] - arrival_intervals[arrival_time_id])
        # total running time required = load cache first and then compute the rest

        # remaining loading time of cache
        prefill_cache_loading_time_remained = round_2(config.ongoing_request.prompt_length * config.ongoing_request.args.cache_loading_speed * (config.ongoing_request.prefill_cache_proportion-config.ongoing_request.prefill_cache_loaded_proportion))
        decoding_cache_loading_time_remained = round_2(config.ongoing_request.args.cache_loading_speed * (config.ongoing_request.decoding_cache_length-config.ongoing_request.decoding_cache_loaded_length))

        full_time_required = config.ongoing_request.previous_save_time + prefill_cache_loading_time_remained + decoding_cache_loading_time_remained + config.ongoing_request.remaining_computation_time[-1]

        print("*"*50)
        print('running_time_for_this_interval:', running_time_for_this_interval)
        print('full_time_required:', full_time_required)
        print("*"*50)
        # if the running time for this interval is longer than the current request
        # we just keep putting the next highest priority request to the GPU while at the same time computing the next highest priority request
        # until the running time for this interval is used up
        # otherwise, we just run the current request and compute the next highest priority request
        if running_time_for_this_interval > full_time_required:
            while running_time_for_this_interval > full_time_required and (len(full_queue.two_dim_priority_queue.heap)+ len(full_queue.unsorted_nodes)) !=0:
                # put the currently highest priority user request to the GPU
                # update both actual time and predicted time
                config.ongoing_request.update_computation_time_normal_run(full_time_required)
                full_queue.update_waiting_time(full_time_required)
                config.ongoing_request.update_waiting_time(full_time_required)
 
                # update log
                config.record_requests[config.ongoing_request.user_request_id] = config.ongoing_request
                config.ongoing_request.print_out_features()
                full_queue.visualize()
                print('-'*50)
                print('-'*50)
                await asyncio.sleep(full_time_required)
                
                # running time for the next request
                running_time_for_this_interval -= full_time_required
                ##### write one function to compute the next highest priority request asynchronously #####
                # put the currently highest priority user request to the GPU
                config.ongoing_request = full_queue.fetch_next_node()
                ##### write one function to compute the next highest priority request asynchronously #####

                # need to load cache first and then compute the rest
                prefill_cache_loading_time_remained = round_2(config.ongoing_request.prompt_length * config.ongoing_request.args.cache_loading_speed * (config.ongoing_request.prefill_cache_proportion-config.ongoing_request.prefill_cache_loaded_proportion))
                decoding_cache_loading_time_remained = round_2(config.ongoing_request.args.cache_loading_speed * (config.ongoing_request.decoding_cache_length-config.ongoing_request.decoding_cache_loaded_length))
                full_time_required = config.ongoing_request.previous_save_time + prefill_cache_loading_time_remained + decoding_cache_loading_time_remained +  + config.ongoing_request.remaining_computation_time[-1]

                # if we run out of the running time for this interval
                if running_time_for_this_interval <= full_time_required:
                    # asynchronously compute the next node
                    # put the currently highest priority user request to the GPU
                    config.ongoing_request.update_computation_time_normal_run(running_time_for_this_interval)
                    full_queue.update_waiting_time(running_time_for_this_interval)
                    config.ongoing_request.update_waiting_time(running_time_for_this_interval)
                    config.record_requests[config.ongoing_request.user_request_id] = config.ongoing_request

                    config.ongoing_request.print_out_features()
                    full_queue.visualize()
                    print('-'*50)
                    print('-'*50)
                    await asyncio.sleep(running_time_for_this_interval)

                elif (len(full_queue.two_dim_priority_queue.heap) + len(full_queue.unsorted_nodes)) == 0:
                    # if there is no more request in the queue, and the ongoing request is also completed
                    # put the currently highest priority user request to the GPU
                    config.ongoing_request.update_computation_time_normal_run(config.ongoing_request.remaining_computation_time[-1])
                    config.ongoing_request.update_waiting_time(config.ongoing_request.remaining_computation_time[-1])
                    
                    config.record_requests[config.ongoing_request.user_request_id] = config.ongoing_request
                    config.ongoing_request.print_out_features()
                    config.ongoing_request = None
                    full_queue.visualize()
                    print('-'*50)
                    print('-'*50)
                    
                    await asyncio.sleep(running_time_for_this_interval)
                    break

        else:
            # run the currently highest priority user request on the GPU
            config.ongoing_request.update_computation_time_normal_run(running_time_for_this_interval)
            full_queue.update_waiting_time(running_time_for_this_interval)
            config.ongoing_request.update_waiting_time(running_time_for_this_interval)

            config.record_requests[config.ongoing_request.user_request_id] = config.ongoing_request
            config.ongoing_request.print_out_features()
            full_queue.visualize()
            print('-'*50)
            print('-'*50)
            
            await asyncio.sleep(running_time_for_this_interval)
    
    else:
        print(bcolors.WARNING + 'Last batch of requests' + bcolors.ENDC)
        # after we have put all requests in the queue, we just process 1 by 1
        while (len(full_queue.two_dim_priority_queue.heap) + len(full_queue.unsorted_nodes)) != 0:
            # remaining loading time of cache
            prefill_cache_loading_time_remained = round_2(config.ongoing_request.prompt_length * config.ongoing_request.args.cache_loading_speed * (config.ongoing_request.prefill_cache_proportion-config.ongoing_request.prefill_cache_loaded_proportion))
            decoding_cache_loading_time_remained = round_2(config.ongoing_request.args.cache_loading_speed * (config.ongoing_request.decoding_cache_length-config.ongoing_request.decoding_cache_loaded_length))
            running_time_for_this_interval = config.ongoing_request.previous_save_time + prefill_cache_loading_time_remained + decoding_cache_loading_time_remained + config.ongoing_request.remaining_computation_time[-1]

            # put the currently highest priority user request to the GPU
            config.ongoing_request.update_computation_time_normal_run(running_time_for_this_interval)
            full_queue.update_waiting_time(running_time_for_this_interval)
            config.ongoing_request.update_waiting_time(running_time_for_this_interval)
            config.record_requests[config.ongoing_request.user_request_id] = config.ongoing_request
            config.ongoing_request.print_out_features()
            full_queue.visualize()
            print('-'*50)
            print('-'*50)
            
            await asyncio.sleep(running_time_for_this_interval)

            config.ongoing_request = full_queue.fetch_next_node()

            if (len(full_queue.two_dim_priority_queue.heap) + len(full_queue.unsorted_nodes)) == 0:
                # remaining loading time of cache
                prefill_cache_loading_time_remained = round_2(config.ongoing_request.prompt_length * config.ongoing_request.args.cache_loading_speed * (config.ongoing_request.prefill_cache_proportion-config.ongoing_request.prefill_cache_loaded_proportion))
                decoding_cache_loading_time_remained = round_2(config.ongoing_request.args.cache_loading_speed * (config.ongoing_request.decoding_cache_length-config.ongoing_request.decoding_cache_loaded_length))
                running_time_for_this_interval = config.ongoing_request.previous_save_time + prefill_cache_loading_time_remained + decoding_cache_loading_time_remained + config.ongoing_request.remaining_computation_time[-1]
            
                config.ongoing_request.update_computation_time_normal_run(running_time_for_this_interval)
                config.ongoing_request.update_waiting_time(running_time_for_this_interval)
                config.record_requests[config.ongoing_request.user_request_id] = config.ongoing_request
                config.ongoing_request.print_out_features()
                full_queue.visualize()
                print('-'*50)
                print('-'*50)
                
                await asyncio.sleep(running_time_for_this_interval)

    full_queue.is_GPU_job_completed = True 

async def simulator(args, user_requests):
    user_request_waiting_time_from_predictor, user_request_arrival_time_from_predictor, time_arrived_requests = compute_arrival_time(args, user_requests)     
    
    print('requests arrive at specific timestamp:', time_arrived_requests)

    arrival_intervals = list(time_arrived_requests.keys())
    oracle_predicted_priority = oracle_priority_predictor(args, list(range(args.user_request_num)))
    oracle_predicted_output_bucket = oracle_output_length_bucket_predictor(args, list(range(args.user_request_num)))
    simulated_predicted_priority = simulated_priority_predictor(args, oracle_predicted_priority)
    simulated_output_length_bucket = simulated_output_length_bucket_predictor(args, oracle_predicted_output_bucket)

    # Create the priority queue for user request queue management
    full_queue = PriorityQueue(args)
    config.record_requests = {}

    config.ongoing_request = None
    for arrival_time_id, arrived_user_request_ids in enumerate(list(time_arrived_requests.values())):
        # add incoming requests to the queue
        # and get the next node to process --> max operation
        for user_request_id in arrived_user_request_ids:
            user_request = Request(args, user_request_id, simulated_predicted_priority[user_request_id], simulated_output_length_bucket[user_request_id])
            full_queue.add_unsorted_node(user_request)

        asyncio.create_task(full_queue.incremental_update(), name=f'incremental_update_task_{arrival_time_id}')

        if config.ongoing_request is None:
            # fetch the next node
            config.ongoing_request = full_queue.fetch_next_node()
        else:
            # fetch the highest ordered request in the newly incoming batch of requests
            next_request_in_batch = full_queue.fetch_next_node()

            # preempt the ongoing request by the new request if necessary
            if trigger_preemption(args, config.ongoing_request, next_request_in_batch):
                print(bcolors.WARNING + f"Preempting {config.ongoing_request.user_request_id} by {next_request_in_batch.user_request_id}" + bcolors.ENDC)
                
                next_request_in_batch.print_out_features()

                # add loading_cache/recomputation time to total remaining time of the preempted request
                ongoing_request_cache_save_time = update_cache_loading_or_recomputation_time_and_extra_saving_time(args, config.ongoing_request)

                # put the preempted request back to the queue
                full_queue.two_dim_priority_queue.push(config.ongoing_request, config.ongoing_request.predicted_priority, config.ongoing_request.predicted_remaining_computation_time[-1] + config.ongoing_request.prefill_cache_loading_time + config.ongoing_request.decoding_cache_loading_time)

                # next time, we need to reload the cache
                config.ongoing_request.prefill_cache_loaded_proportion = 0
                config.ongoing_request.decoding_cache_loaded_length = 0

                # the preempting request is now the ongoing request
                config.ongoing_request = next_request_in_batch
                # we also consider the situation where the previous saving time is not 0
                # meaning that the request before the preempted request has not finished saving the cache
                # in which case ongoing_request_cache_save_time = 0, but we need to add the previous saving time to the current request
                config.ongoing_request.previous_save_time = ongoing_request_cache_save_time + config.ongoing_request.previous_save_time
                
                full_queue.visualize()
            else:
                print(bcolors.WARNING + "No preemption" + bcolors.ENDC)
                # add next_request_in_batch in queue right after the root
                full_queue.two_dim_priority_queue.push(next_request_in_batch, config.ongoing_request.predicted_priority, next_request_in_batch.predicted_remaining_computation_time[-1] + next_request_in_batch.prefill_cache_loading_time + config.ongoing_request.decoding_cache_loading_time)
        
        if arrival_time_id < len(arrival_intervals) - 1:
            running_time = round_2(arrival_intervals[arrival_time_id+1] - arrival_intervals[arrival_time_id])
            asyncio.create_task(GPU_execute(arrival_time_id, arrival_intervals, full_queue, config.ongoing_request, config.record_requests), name=f'GPU_execute_task_{config.ongoing_request.user_request_id}')
            await asyncio.sleep(running_time)
        else:
            await GPU_execute(arrival_time_id, arrival_intervals, full_queue, config.ongoing_request, config.record_requests)

    save_info, file_name = compute_average_waiting_time(args, config.record_requests)

    return save_info, file_name

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gap_num', type=int, default=10)

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

    parser.add_argument('--token_decoding_speed_coefficient1', type=float, default=2.86938135e-06)
    parser.add_argument('--token_decoding_speed_coefficient2', type=float, default=3.00834536e-02)
    parser.add_argument('--prefill_speed_coefficient1', type=float, default=1.58625005e-09)
    parser.add_argument('--prefill_speed_coefficient2', type=float, default=1.30542054e-04)
    parser.add_argument('--cache_loading_speed', type=float, default=0.0003, help='cache loading speed per token')
    parser.add_argument('--cache_saving_speed', type=float, default=0.0003, help='cache saving speed per token')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

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
