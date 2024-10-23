import argparse
import numpy as np
import random
import config
import time
from queue_management import Request


def initialize_user_request(args):
    user_request_priority = [random.choice(list(range(args.ranks))) for _ in range(args.user_request_num)]
    user_request_prompt_length = [random.choice(list(range(1, args.max_prompt_length))) for _ in range(args.user_request_num)]
    user_request_output_length = [random.choice(list(range(1, args.max_output_length))) for user_request in user_request_prompt_length]
    return user_request_priority, user_request_prompt_length, user_request_output_length


def simulator(args):
    # Create the priority queue for user request queue management
    queue = []

    total_time = 0

    # Initialize the running time of the first request
    ongoing_request_id = 0
    ongoing_request = Request(args, ongoing_request_id)
    queue.append(ongoing_request)
    # upcoming requests 
    for new_user_request_id in range(1, args.user_request_num):
        # Update the running time of currently running request
        ongoing_request.update_computation_time_normal_run(args.user_request_gap)
        ongoing_request.print_out_features()

        # add new user request to the queue
        new_user_request = Request(args, new_user_request_id)
        queue.append(new_user_request)
        new_user_request.print_out_features()

        # if ongoing request finishes
        if ongoing_request.remaining_computation_time[-1] <= 0:
            queue = queue[1:]
            # put the next user request to the GPU
            new_ongoing_request = queue[0]
            new_ongoing_request.update_computation_time_normal_run(args.user_request_gap + ongoing_request.remaining_computation_time[-1])
            ongoing_request = new_ongoing_request

        total_time += args.user_request_gap

        print('-'*50)
        print('-'*50)
        time.sleep(0.1)

    print('---'*50)
    for node in queue:
        node.print_out_features()
    for user_request in queue:
        total_time += user_request.remaining_computation_time[-1]

    print(f'Total time: {total_time}')


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

    simulator(args)
