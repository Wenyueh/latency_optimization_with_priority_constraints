import argparse
import numpy as np
import random
import config
import time
from queue_management import Request, PriorityQueue
from utils import bcolors
from semantic_predictor import oracle_priority_predictor, oracle_output_length_bucket_predictor, compute_total_generation_time, cache_loading_time, cache_saving_time


def initialize_user_request(args):
    user_request_priority = [random.choice(list(range(args.ranks))) for _ in range(args.user_request_num)]
    user_request_prompt_length = [random.choice(list(range(1, args.max_prompt_length))) for _ in range(args.user_request_num)]
    user_request_output_length = [random.choice(list(range(1, args.max_output_length))) for user_request in user_request_prompt_length]
    return user_request_priority, user_request_prompt_length, user_request_output_length

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


def simulator(args):
    # record total running time
    total_time = 0

    # Create the priority queue for user request queue management
    queue = PriorityQueue(args)

    # Initialize the running time of the first request
    ongoing_request_id = 0
    ongoing_request = Request(args, ongoing_request_id)
    # upcoming requests 
    for new_user_request_id in range(1, args.user_request_num):
        # Update the running time of currently running request
        ongoing_request.update_computation_time_normal_run(args.user_request_gap)
        ongoing_request.update_predicted_computation_time_normal_run(args.user_request_gap)

        ongoing_request.print_out_features()

        new_user_request = Request(args, new_user_request_id)
        new_user_request.print_out_features()

        print('**** check preemption ****')
        print('**************************')

        # preempt the ongoing request by the new request if necessary
        if trigger_preemption(ongoing_request, new_user_request):
            print(bcolors.WARNING + f"Preempting {ongoing_request.user_request_id} by {new_user_request.user_request_id}" + bcolors.ENDC)
            ongoing_request.update_computation_time_preempted()
            ongoing_request.update_predicted_computation_time_preempted()

            ongoing_request.print_out_features()

            queue.insert_node(ongoing_request_id)

            ongoing_request = new_user_request
            ongoing_request_id = new_user_request_id
        else:
            print(bcolors.WARNING + "No preemption" + bcolors.ENDC)
            queue.insert_node(new_user_request_id)

            # Get the highest priority request to process
            highest_priority_request_id = queue.get_highest_priority_request_id()
            highest_priority_request = queue.nodes[highest_priority_request_id]

        # if ongoing request finishes
        if ongoing_request.remaining_computation_time[-1] <= 0:
            print("------PAY ATTENTION: ONGOING REQUEST FINISHES------")
            # put the currently highest priority user request to the GPU
            highest_priority_request.update_computation_time_normal_run(args.user_request_gap + ongoing_request.remaining_computation_time[-1])
            highest_priority_request.update_predicted_computation_time_normal_run(args.user_request_gap + ongoing_request.remaining_computation_time[-1])
            highest_priority_request.print_out_features()

            # the highest priority request is now the ongoing request
            ongoing_request_id = highest_priority_request_id
            ongoing_request = highest_priority_request

            # remove the highest priority request from the queue
            queue.remove_node(ongoing_request_id)

            # Update the tree for the current highest priority request
            highest_priority_request_id = queue.get_highest_priority_request_id()
            highest_priority_request = queue.nodes[highest_priority_request_id]

        queue.visualize(queue.root)
        print('-'*50)
        print('-'*50)
        time.sleep(0.1)

        total_time += args.user_request_gap

    total_time += ongoing_request.remaining_computation_time[-1]
    print('---'*50)

    while len(queue.nodes) > 0:
        highest_priority_request_id = queue.get_highest_priority_request_id()
        highest_priority_request = queue.nodes[highest_priority_request_id]
        highest_priority_request.print_out_features()
        # finish the remaining requests
        total_time += highest_priority_request.remaining_computation_time[-1]

        # remove the highest priority request from the queue
        queue.remove_node(highest_priority_request_id)

    print(f"Total time: {total_time}")



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

    random.seed(args.seed)
    np.random.seed(args.seed)

    config.user_request_priority, config.user_request_prompt_length, config.user_request_output_length = initialize_user_request(args)

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

    simulator(args)
