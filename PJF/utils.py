import asyncio, time, json, os
from async_timeout import timeout
import numpy as np
    

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def round_2(some_input):
    if isinstance(some_input, float):
        return round(some_input,10)
    if isinstance(some_input, list):
        return [round_2(x) for x in some_input]
    if isinstance(some_input, int):
        return some_input
    

async def cancel(task):
    start = time.time()
    task.cancel()
    try:
        async with timeout(-1):
            await task
        end = time.time()
    except asyncio.CancelledError:
        end = time.time()
        return 'task cancelled'
    except asyncio.exceptions.TimeoutError:
        end = time.time()
        return 'time out for canceling task'
    except Exception as e:
        end = time.time()
        return 'exception:' + str(e)
    

def is_sorted(list_of_requests):
    """
    Check if the list of requests is sorted by waiting time
    """
    for i in range(len(list_of_requests)-1):
        if not(list_of_requests[i].predicted_priority < list_of_requests[i+1].predicted_priority or (list_of_requests[i].predicted_priority == list_of_requests[i+1].predicted_priority and list_of_requests[i].predicted_remaining_computation_time[-1] + list_of_requests[i].prefill_cache_loading_time + list_of_requests[i].decoding_cache_loading_time < list_of_requests[i+1].predicted_remaining_computation_time[-1] + list_of_requests[i+1].prefill_cache_loading_time + list_of_requests[i+1].decoding_cache_loading_time)): 
            return False
    return True
    

def compute_waiting_time_with_priority(record_requests, user_request_waiting_time_from_predictor):
    save_info = {}
    waiting_time_with_priority = {}
    for i in range(len(record_requests)):
        try:
            save_info[i] = {'user_request_id': record_requests[i].user_request_id, 'waiting_time': record_requests[i].waiting_time+user_request_waiting_time_from_predictor[i], 'priority': record_requests[i].priority, 'predicted_priority': record_requests[i].predicted_priority, 'prompt_length': record_requests[i].prompt_length, 'output_length': record_requests[i].output_length, 'predicted_output_length': record_requests[i].predicted_output_length}
            print(f"User request {record_requests[i].user_request_id} has a total waiting time of {record_requests[i].waiting_time}")
            if record_requests[i].priority not in waiting_time_with_priority:
                waiting_time_with_priority[record_requests[i].priority] = [record_requests[i].waiting_time + user_request_waiting_time_from_predictor[i]]
            else:
                waiting_time_with_priority[record_requests[i].priority].append(record_requests[i].waiting_time + user_request_waiting_time_from_predictor[i])
        except Exception as e:
            print(f"Error in computing waiting time for user request {record_requests[i].user_request_id}: {e}")
            print(record_requests[i].priority)
            print(record_requests[i].waiting_time)
            print(record_requests[i].predicted_remaining_time)
            continue
    for k,v in waiting_time_with_priority.items():
        print(f"Priority {k} has an average waiting time of {np.mean(v)}")
        waiting_time_with_priority[k] = round_2(np.mean(v))

    # sort waiting time with priority by key from small to large
    waiting_time_with_priority = dict(sorted(waiting_time_with_priority.items()))
    save_info['average_waiting_time_with_priority'] = waiting_time_with_priority

    return save_info, waiting_time_with_priority
    

def compute_average_waiting_time(args, record_requests, user_request_waiting_time_from_predictor):
    save_info, _ = compute_waiting_time_with_priority(record_requests, user_request_waiting_time_from_predictor)
    priority_predictor_latency = round(args.priority_predictor_latency/(args.priority_predictor_batching_size*0.1), 4)

    if args.experiment_name == 'gap':
        n = 0
        file_name = f'simulation/gap/average_waiting_time_with_priority_gap{args.user_request_gap}_maxcon{args.max_concurrent_user_requests}_{n}.json'
        while os.path.exists(file_name):
            n += 1
            file_name = f'simulation/gap/average_waiting_time_with_priority_gap{args.user_request_gap}_maxcon{args.max_concurrent_user_requests}_{n}.json'

    elif args.experiment_name == 'priority':
        n = 0
        file_name = f'simulation/priority/average_waiting_time_with_priority_priorityrate{args.priority_error_rate}_dis{args.priority_error_distance}_{n}.json'
        while os.path.exists(file_name):
            n += 1
            file_name = f'simulation/priority/average_waiting_time_with_priority_priorityrate{args.priority_error_rate}_dis{args.priority_error_distance}_{n}.json'

    elif args.experiment_name == 'length':
        n = 0
        file_name = f'simulation/length/average_waiting_time_with_priority_lengthrate{args.output_length_error_rate}_dis{args.output_length_error_distance}_{n}.json'
        while os.path.exists(file_name):
            n += 1
            file_name = f'simulation/length/average_waiting_time_with_priority_lengthrate{args.output_length_error_rate}_dis{args.output_length_error_distance}_{n}.json'
    else:
        assert args.experiment_name == 'latency'
        n = 0
        file_name = f'simulation/latency/average_waiting_time_with_priority_predictorbatch{args.priority_predictor_batching_size}_latency{priority_predictor_latency}_{args.processing_mode}_{n}.json'
        while os.path.exists(file_name):
            n += 1
            file_name = f'simulation/latency/average_waiting_time_with_priority_predictorbatch{args.priority_predictor_batching_size}_latency{priority_predictor_latency}_{args.processing_mode}_{n}.json'
        
    
    return save_info, file_name
