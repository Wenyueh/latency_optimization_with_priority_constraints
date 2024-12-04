import argparse
import numpy as np
import random
import config
import time
from queue_management import Request, PriorityQueue
from utils import bcolors
from semantic_predictor import oracle_priority_predictor, oracle_output_length_bucket_predictor, compute_total_generation_time, cache_loading_time, cache_saving_time
import asyncio
import math

def initialize_user_request(args):
    user_request_priority = [random.choice(list(range(args.ranks))) for _ in range(args.user_request_num)]
    user_request_prompt_length = [random.choice(list(range(1, args.max_prompt_length))) for _ in range(args.user_request_num)]
    user_request_output_length = [random.choice(list(range(1, args.max_output_length))) for user_request in user_request_prompt_length]
    return user_request_priority, user_request_prompt_length, user_request_output_length

## need to rewrite the trigger preemption function!!
## need to rewrite the trigger preemption function!!
## need to rewrite the trigger preemption function!!
def trigger_preemption(user_request_1_object, user_request_2_object):
    # do we want to preempt user_request_1 by user_request_2
    def predicted_running_time_if_preempted(user_request_object):
        _, _, complete_running_time = user_request_object.predicted_remaining_computation_time
        cache_load_time = cache_loading_time(args, user_request_object.user_request_id, user_request_object.finished_computation_time)
        cache_save_time = cache_saving_time(args, user_request_object.user_request_id, user_request_object.finished_computation_time)
        return cache_save_time, cache_load_time + complete_running_time
    
    def predicted_running_time_if_preempting(user_request_object):
        _, _, complete_running_time = user_request_object.predicted_remaining_computation_time
        cache_load_time = cache_loading_time(args, user_request_object.user_request_id, user_request_object.finished_computation_time)
        return cache_load_time + complete_running_time
        
    if user_request_1_object.predicted_priority > user_request_2_object.predicted_priority:
        return True 
    elif user_request_1_object.predicted_priority == user_request_2_object.predicted_priority:
       cache_save_time, preempted_remaining_time = predicted_running_time_if_preempted(user_request_1_object)
       if preempted_remaining_time > cache_save_time + predicted_running_time_if_preempting(user_request_2_object):
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


async def simulator(args, user_requests):
    user_request_waiting_time_from_predictor, user_request_arrival_time_from_predictor, time_arrived_requests = compute_arrival_time(args, user_requests)     
    
    print('requests arrive at specific timestamp:', time_arrived_requests)

    arrival_intervals = list(time_arrived_requests.keys())
    oracle_predicted_priority = oracle_priority_predictor(list(range(args.user_request_num)))
    oracle_predicted_output_length = await oracle_output_length_bucket_predictor(args, list(range(args.user_request_num)))

    # Create the priority queue for user request queue management
    full_queue = PriorityQueue(args)
    record_requests = full_queue.nodes.copy()

    ongoing_request = None
    for arrival_time_id, arrived_user_request_ids in enumerate(list(time_arrived_requests.values())):
        # if there is no ongoing request
        # in which case the queue is definitely empty
        # we just put the highest priority request to the GPU
        if ongoing_request is None:
            # add incoming requests to the queue
            for request_id in arrived_user_request_ids:
                new_user_request = Request(args, request_id, oracle_predicted_priority[request_id], oracle_predicted_output_length[request_id])
                full_queue.insert_node(new_user_request)
            # get the next node to process
            await full_queue.compute_first_node()
            # fetch the next node
            ongoing_request = full_queue.fetch_next_node()
        else:
            # build a temp queue
            temp_queue = PriorityQueue(args)
            # add incoming requests to the queue
            for request_id in arrived_user_request_ids:
                request = Request(args, request_id, oracle_predicted_priority[request_id], oracle_predicted_output_length[request_id])
                temp_queue.insert_node(request)

            # fetch the highest ordered request in the newly incoming batch of requests
            # get the next node to process
            await temp_queue.compute_first_node()
            # fetch the next node
            next_request_in_batch = temp_queue.fetch_next_node()

            # preempt the ongoing request by the new request if necessary
            if trigger_preemption(ongoing_request, next_request_in_batch):
                print(bcolors.WARNING + f"Preempting {ongoing_request.user_request_id} by {next_request_in_batch.user_request_id}" + bcolors.ENDC)
                ongoing_request.update_computation_time_preempted()
                ongoing_request.update_predicted_computation_time_preempted()

                # put the preempted request back to the queue
                full_queue.insert_node(ongoing_request)
                root_children_ids = list(full_queue.root.children.keys())
                root_children_ids.remove(ongoing_request.user_request_id)
                for existent_request_id in root_children_ids:
                    full_queue.add_dependency(ongoing_request.user_request_id, existent_request_id)

                # the preempting request is now the ongoing request
                ongoing_request = next_request_in_batch

                # add the rest of the requests to the queue, right after the root node
                for new_request in temp_queue.nodes.values():
                    if new_request.user_request_id != ongoing_request.user_request_id:
                        full_queue.insert_node(new_request)

            else:
                print(bcolors.WARNING + "No preemption" + bcolors.ENDC)
                # add next_request_in_batch in queue right after the root
                full_queue.insert_node(next_request_in_batch)
                # add the rest of the requests to the queue after next_request_in_batch
                for new_request in temp_queue.nodes.values():
                    if new_request.user_request_id != next_request_in_batch.user_request_id:
                        full_queue.insert_node(new_request)
                        full_queue.add_dependency(next_request_in_batch.user_request_id, new_request.user_request_id)
        
        # if it is not the last batch
        if arrival_time_id < len(arrival_intervals) - 1:
            # compute the running time for this interval, before the next batch of requests arrive
            running_time_for_this_interval = round(arrival_intervals[arrival_time_id+1] - arrival_intervals[arrival_time_id],2)
            
            # if the running time for this interval is longer than the current request
            # we just keep putting the next highest priority request to the GPU while at the same time computing the next highest priority request
            # until the running time for this interval is used up
            # otherwise, we just run the current request and compute the next highest priority request
            if running_time_for_this_interval > ongoing_request.remaining_computation_time[-1]:
                while running_time_for_this_interval > ongoing_request.remaining_computation_time[-1] and len(full_queue.nodes) !=0:
                    # asynchronously compute the next node
                    asyncio.create_task(full_queue.compute_first_node(),name=f'find_max_after_{ongoing_request.user_request_id}')
                    await asyncio.sleep(ongoing_request.remaining_computation_time[-1])
                    # put the currently highest priority user request to the GPU
                    ongoing_request.update_computation_time_normal_run(ongoing_request.remaining_computation_time[-1])
                    ongoing_request.update_predicted_computation_time_normal_run(ongoing_request.remaining_computation_time[-1])
                    full_queue.update_waiting_time(ongoing_request.remaining_computation_time[-1])
                    record_requests[ongoing_request.user_request_id] = ongoing_request

                    ############################################
                    ######### print out the features ###########
                    ############################################
                    ongoing_request.print_out_features()
                    full_queue.visualize(full_queue.root)
                    print('-'*50)
                    print('-'*50)
                    
                    # running time for the next request
                    running_time_for_this_interval -= ongoing_request.remaining_computation_time[-1]
                    # put the currently highest priority user request to the GPU
                    ongoing_request = full_queue.fetch_next_node()

                    # if we run out of the running time for this interval
                    if running_time_for_this_interval <= ongoing_request.remaining_computation_time[-1]:
                        # asynchronously compute the next node
                        if len(full_queue.nodes) != 0:
                            asyncio.create_task(full_queue.compute_first_node(),name=f'find_max_after_{ongoing_request.user_request_id}')
                        await asyncio.sleep(running_time_for_this_interval)
                        # put the currently highest priority user request to the GPU
                        ongoing_request.update_computation_time_normal_run(running_time_for_this_interval)
                        ongoing_request.update_predicted_computation_time_normal_run(running_time_for_this_interval)
                        full_queue.update_waiting_time(running_time_for_this_interval)
                        record_requests[ongoing_request.user_request_id] = ongoing_request

                        ############################################
                        ######### print out the features ###########
                        ############################################
                        ongoing_request.print_out_features()
                        full_queue.visualize(full_queue.root)
                        print('-'*50)
                        print('-'*50)

                    elif len(full_queue.nodes) == 0:
                        await asyncio.sleep(running_time_for_this_interval)
                        # put the currently highest priority user request to the GPU
                        ongoing_request.update_computation_time_normal_run(ongoing_request.remaining_computation_time[-1])
                        ongoing_request.update_predicted_computation_time_normal_run(ongoing_request.remaining_computation_time[-1])
                        ongoing_request = None

                        ############################################
                        ######### print out the features ###########
                        ############################################
                        ongoing_request.print_out_features()
                        full_queue.visualize(full_queue.root)
                        print('-'*50)
                        print('-'*50)
            else:
                # asynchronously compute the next node
                asyncio.create_task(full_queue.compute_first_node(),name=f'find_max_after_{ongoing_request.user_request_id}')
                await asyncio.sleep(running_time_for_this_interval)
                # run the currently highest priority user request on the GPU
                ongoing_request.update_computation_time_normal_run(running_time_for_this_interval)
                ongoing_request.update_predicted_computation_time_normal_run(running_time_for_this_interval)
                full_queue.update_waiting_time(running_time_for_this_interval)
                record_requests[ongoing_request.user_request_id] = ongoing_request

                ############################################
                ######### print out the features ###########
                ############################################
                ongoing_request.print_out_features()
                full_queue.visualize(full_queue.root)
                print('-'*50)
                print('-'*50)
        
        else:
            # after we have put all requests in the queue, we just process 1 by 1
            while len(full_queue.nodes) != 0:
                running_time_for_this_interval = ongoing_request.remaining_computation_time[-1]
                print('running_time_for_this_interval')
                print(running_time_for_this_interval)

                asyncio.create_task(full_queue.compute_first_node(),name=f'find_max_after_{ongoing_request.user_request_id}')
                await asyncio.sleep(running_time_for_this_interval)

                # put the currently highest priority user request to the GPU
                ongoing_request.update_computation_time_normal_run(running_time_for_this_interval)
                ongoing_request.update_predicted_computation_time_normal_run(running_time_for_this_interval)
                full_queue.update_waiting_time(running_time_for_this_interval)
                record_requests[ongoing_request.user_request_id] = ongoing_request

                ############################################
                ######### print out the features ###########
                ############################################
                ongoing_request.print_out_features()
                full_queue.visualize(full_queue.root)
                print('-'*50)
                print('-'*50)

                ongoing_request = full_queue.fetch_next_node()

                if len(full_queue.nodes) == 0:
                    await asyncio.sleep(ongoing_request.remaining_computation_time[-1])
                    ongoing_request.update_computation_time_normal_run(ongoing_request.remaining_computation_time[-1])
                    ongoing_request.update_predicted_computation_time_normal_run(ongoing_request.remaining_computation_time[-1])
                    record_requests[ongoing_request.user_request_id] = ongoing_request

                    ############################################
                    ######### print out the features ###########
                    ############################################
                    ongoing_request.print_out_features()
                    full_queue.visualize(full_queue.root)
                    print('-'*50)
                    print('-'*50)

    for i in range(len(record_requests)):
        record_requests[i].waiting_time += record_requests[i].finished_computation_time + user_request_waiting_time_from_predictor[i]
        print(f"User request {record_requests[i].user_request_id} has a total waiting time of {record_requests[i].waiting_time}")

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

    parser.add_argument('--gap_num', type=int, default=5)
    parser.add_argument('--ranks', type=int, default=10)
    parser.add_argument('--length_bucket_num', type=int, default=20)
    parser.add_argument('--max_prompt_length', type=int, default=20)
    parser.add_argument('--max_output_length', type=int, default=100)

    parser.add_argument('--token_decoding_speed', type=float, default=0.05)
    parser.add_argument('--prefill_speed', type=float, default=0.05)
    parser.add_argument('--cache_loading_speed', type=float, default=0.01, help='cache loading speed per token')
    parser.add_argument('--cache_saving_speed', type=float, default=0.01, help='cache saving speed per token')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # for each time interval, we have a random number of user requests coming
    num_of_user_requests_coming = [random.randint(0, 3) for _ in range(args.gap_num)]
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