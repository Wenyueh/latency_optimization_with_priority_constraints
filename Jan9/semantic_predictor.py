import config, random, argparse
import math
from utils import round_2
import random, time
import numpy as np
from tqdm import tqdm

def initialize_user_request(args):
    user_request_priority = [random.choice(list(range(args.ranks))) for _ in range(args.user_request_num)]
    user_request_prompt_length = [random.choice(list(range(1, args.max_prompt_length))) for _ in range(args.user_request_num)]
    user_request_output_length = [random.choice(list(range(1, args.max_output_length))) for user_request in user_request_prompt_length]
    return user_request_priority, user_request_prompt_length, user_request_output_length


# simulated oracle predictor
def oracle_priority_predictor(args, user_request_ids):
    priorities = [config.user_request_priority[user_request_id] for user_request_id in user_request_ids]
    return priorities

def oracle_output_length_bucket_predictor(args, user_request_ids):
    r = random.randint(0,1)
    if r == 1:
        output_length_buckets = [
            math.ceil(config.user_request_output_length[user_request_id] / 
            (args.max_output_length/args.length_bucket_num))
            for user_request_id in user_request_ids
            ]
    else:
        output_length_buckets = [
            math.floor(config.user_request_output_length[user_request_id] / 
            (args.max_output_length/args.length_bucket_num))
            for user_request_id in user_request_ids
            ]
    return output_length_buckets

# % of user requests that are mistaken by an error rate of priority_error_rate
# mistaken user requests have a priority that is within priority_error_distance of the true priority
def simulated_priority_predictor(args, priorities):
    user_request_ids = list(range(len(priorities)))
    mistaken_user_request_ids = random.sample(user_request_ids, int(len(user_request_ids)*args.priority_error_rate))
    for mistaken_user_request_id in mistaken_user_request_ids:
        error_distance = int(args.ranks * args.priority_error_distance)
        mistaken_priority = min(args.ranks, max(1, config.user_request_priority[mistaken_user_request_id] + random.randint(-error_distance+1, error_distance+1)))
        priorities[mistaken_user_request_id] = mistaken_priority
    return priorities

def simulated_output_length_bucket_predictor(args, output_length_buckets):
    user_request_ids = list(range(len(output_length_buckets)))
    mistaken_user_request_ids = random.sample(user_request_ids, int(len(user_request_ids)*args.output_length_error_rate))
    for mistaken_user_request_id in mistaken_user_request_ids:
        length_bucket_error_distance = int(args.length_bucket_num * args.output_length_error_distance)
        mistaken_bucket = min(args.length_bucket_num, max(1, output_length_buckets[mistaken_user_request_id] + random.randint(-length_bucket_error_distance+1, length_bucket_error_distance+1)))
        output_length_buckets[mistaken_user_request_id] = mistaken_bucket
    return output_length_buckets

# helper functions
# relationship between computation time & length in prompt/output
# every token O(n) time, sum(prompt_length, prompt_length+output_length) = 1/2 * (prompt_length + output_length) * (prompt_length + output_length + 1)
def compute_output_time(args, prompt_length, output_length):
    p = prompt_length
    o = output_length
    output_time = args.token_decoding_speed_coefficient1 * (1/2 * o**2 + p*o + 1/2 * o) + args.token_decoding_speed_coefficient2 * o
    return round_2(output_time)
# prompt_length ^ 2 * prefill_speed
def compute_prefill_time(args, prompt_length):
    p = prompt_length
    return round_2((p ** 2)*args.prefill_speed_coefficient1 + p*args.prefill_speed_coefficient2)
# given computation length, how many tokens are computed
def compute_generated_tokens(args, prompt_length, output_time):
    a = args.token_decoding_speed_coefficient1/2
    b = args.token_decoding_speed_coefficient1 * (prompt_length + 1/2) + args.token_decoding_speed_coefficient2
    c = args.token_decoding_speed_coefficient1 - output_time
    output_length_possibility_1 = (1/(2*a)) * (-1 * b + math.sqrt(b**2 - 4*a*c))
    output_length_possibility_2 = (1/(2*a)) * (-1 * b - math.sqrt(b**2 - 4*a*c))
    output_length = int(max(output_length_possibility_1, output_length_possibility_2))
    return output_length
def compute_optimal_prefill_length(args, prompt_length):
    return max(0,(args.cache_loading_speed - args.prefill_speed_coefficient2)/(2*args.prefill_speed_coefficient1*prompt_length)) 
def compute_optimal_decoding_length(args, prompt_length):
    part1 = args.token_decoding_speed_coefficient1*prompt_length
    part2 = (args.token_decoding_speed_coefficient1 + 2*args.token_decoding_speed_coefficient2)/2
    optimal = (args.cache_loading_speed - part1 - part2)/(2*args.token_decoding_speed_coefficient1)
    return max(0, int(optimal))

def compute_total_generation_time(args, user_request):
    prompt_length = user_request.prompt_length
    output_length = user_request.output_length
    prefill_time = compute_prefill_time(args, prompt_length)
    output_time = compute_output_time(args, prompt_length, output_length)
    total_generation_time = prefill_time + output_time
    return round_2([prefill_time, output_time, total_generation_time])
def compute_predicted_total_generation_time(args, user_request):
    prompt_length = user_request.prompt_length
    output_length = user_request.predicted_output_length
    prefill_time = compute_prefill_time(args, prompt_length)
    output_time = compute_output_time(args, prompt_length, output_length)
    total_generation_time = prefill_time + output_time
    return round_2([prefill_time, output_time, total_generation_time])

# def cache_decode_or_recompute(args, output_length):
#     cache_length = output_length - math.floor(args.cache_loading_speed/args.token_decoding_speed)
#     recompute_length = output_length - cache_length
#     return cache_length, recompute_length
# def cache_prefill_or_not(args, prompt_length):
#     if args.cache_loading_speed > args.prefill_speed * prompt_length:
#         return False 
#     else:
#         return True

# this is only called after the user request is preempted
def update_cache_loading_or_recomputation_time_and_extra_saving_time(args, user_request):
    finished_computation_time = user_request.finished_computation_time
    # next time, need to reload the cache
    user_request.prefill_cache_loaded_proportion = 0
    user_request.decoding_cache_loaded_length = 0

    prompt_length = user_request.prompt_length
    prefill_time, _, _ = user_request.remaining_computation_time
    
    # finish prefilling and now in decoding phase
    if prefill_time <= 0:
        ######## decoding update ########
        # finished_computation_time - compute_prefill_time(args, prompt_length) is the total time spent on decoding
        total_generated_tokens = compute_generated_tokens(args, prompt_length, finished_computation_time-compute_prefill_time(args, prompt_length))
        # compute the total number of decoded tokens that should be cached
        decoding_cache_length = min(total_generated_tokens, user_request.optimal_decoding_cache_length)
        # the number of tokens that need to be recomputed
        decoding_recompute_length = total_generated_tokens - decoding_cache_length
        # compute the time required for decoding the cache
        decoding_recompute_time = compute_output_time(args, prompt_length+decoding_cache_length, decoding_recompute_length)
        # compute the time spent on saving the new extra cache
        decoding_newly_saving_cache_length = decoding_cache_length - user_request.decoding_cache_length
        decoding_cache_saving_time = decoding_newly_saving_cache_length * args.cache_saving_speed
        # update decoding cache 
        user_request.decoding_cache_length = decoding_cache_length
        user_request.decoding_cache_loading_time = user_request.decoding_cache_length * args.cache_loading_speed   

        ######## prefilling update ########
        # compute the time spent on saving the new extra cache
        prefilling_newly_cache_saving_time = (user_request.optimal_prefill_cache_proportion - user_request.prefill_cache_proportion) * prompt_length * args.cache_saving_speed
        total_saving_cache_time = round_2(prefilling_newly_cache_saving_time + decoding_cache_saving_time)
        # update prefill cache
        user_request.prefill_cache_loading_time = args.cache_loading_speed * user_request.optimal_prefill_cache_proportion * prompt_length
        user_request.prefill_cache_proportion = user_request.optimal_prefill_cache_proportion
        # update prefill recompute time
        prefilling_recompute_proportion = 1 - user_request.optimal_prefill_cache_proportion
        prefilling_recompute_time = compute_prefill_time(args, prefilling_recompute_proportion * prompt_length)

        # update running time required
        user_request.finished_computation_time -= (decoding_recompute_time + prefilling_recompute_time)

        user_request.remaining_computation_time[0] += prefilling_recompute_time
        user_request.remaining_computation_time[1] += decoding_recompute_time
        user_request.remaining_computation_time[-1] += (decoding_recompute_time + prefilling_recompute_time)
        user_request.remaining_computation_time = round_2(user_request.remaining_computation_time)

        user_request.predicted_remaining_computation_time[0] += prefilling_recompute_time
        user_request.predicted_remaining_computation_time[1] += decoding_recompute_time
        user_request.predicted_remaining_computation_time[-1] += (decoding_recompute_time + prefilling_recompute_time)
        user_request.predicted_remaining_computation_time = round_2(user_request.predicted_remaining_computation_time)
    
    # still in prefilling phase
    elif prefill_time > 0:
        total_generated_prefill_proportion = (compute_prefill_time(args, prompt_length)-prefill_time)/compute_prefill_time(args, prompt_length)
        should_cache_prefill_proportion = min(user_request.optimal_prefill_cache_proportion, total_generated_prefill_proportion)
        newly_cache_prefill_proportion = should_cache_prefill_proportion - user_request.prefill_cache_proportion
        user_request.prefill_cache_proportion = should_cache_prefill_proportion
        user_request.prefill_cache_loading_time = user_request.prefill_cache_proportion * prompt_length * args.cache_loading_speed
            
        # compute the time spent on saving the new extra cache
        prefilling_newly_cache_saving_time = newly_cache_prefill_proportion * prompt_length * args.cache_saving_speed
        total_saving_cache_time = round_2(prefilling_newly_cache_saving_time)
        # compute the time spent on recompute the prefilling
        recompute_proportion = total_generated_prefill_proportion - should_cache_prefill_proportion
        prefilling_recompute_time = compute_prefill_time(args, recompute_proportion * prompt_length)

        # update running time required
        user_request.finished_computation_time -= prefilling_recompute_time
        
        user_request.remaining_computation_time[0] += prefilling_recompute_time
        user_request.remaining_computation_time[-1] += prefilling_recompute_time
        user_request.remaining_computation_time = round_2(user_request.remaining_computation_time)

        user_request.predicted_remaining_computation_time[0] += prefilling_recompute_time
        user_request.predicted_remaining_computation_time[-1] += prefilling_recompute_time
        user_request.predicted_remaining_computation_time = round_2(user_request.predicted_remaining_computation_time)

    return total_saving_cache_time

# # this is only called after the user request is preempted
# def compute_cache_saving_time(args, user_request):
#     finished_computation_time = user_request.finished_computation_time
#     prompt_length = user_request.prompt_length
#     prefill_time, _, _ = user_request.predicted_remaining_computation_time
#     # if we are in decoding phase
#     if prefill_time == 0:
#         # total generated tokens
#         # compute the cache time for decoding
#         generated_tokens = compute_generated_tokens(args, prompt_length, finished_computation_time-compute_prefill_time(args, prompt_length))
#         total_cache_length = min(generated_tokens, user_request.optimal_decoding_cache_length)
#         newly_saving_cache_length = min(total_cache_length-user_request.decoding_cache_length, 0)
#         decoding_cache_time = newly_saving_cache_length * args.cache_saving_speed

#         # compute cache time for prefilling
#         if user_request.should_cache_prefill:
#             prefilling_cache_time = (1 - user_request.prefill_cache_proportion) * prompt_length * args.cache_saving_speed
#             user_request.prefill_cache_proportion = 1
#             cache_time = round_2(prefilling_cache_time + decoding_cache_time)
#             return cache_time
#         else:
#             return decoding_cache_time
#     # if we are still in prefilling phase
#     else:
#         if user_request.should_cache_prefill:
#             newly_saving_cache_proportion = finished_computation_time/compute_prefill_time(args, prompt_length) - user_request.prefill_cache_proportion
#             user_request.prefill_cache_proportion = finished_computation_time/prefill_time
#             return round_2(newly_saving_cache_proportion * prompt_length * args.cache_saving_speed)
#         else:
#             return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--priority_error_distance', type=float, default=0.1)
    parser.add_argument('--priority_error_rate', type=float, default=0.1)
    parser.add_argument('--output_length_error_distance', type=float, default=0.1)
    parser.add_argument('--output_length_error_rate', type=float, default=0.1)

    parser.add_argument('--user_request_gap', type=float, default=0.5)
    parser.add_argument('--priority_predictor_batching_size', type=int, default=2)
    parser.add_argument('--priority_predictor_latency', type=float, default=0.003)
    parser.add_argument('--length_predictor_batching_size', type=int, default=2)
    parser.add_argument('--length_predictor_latency', type=float, default=0.003)

    parser.add_argument('--gap_num', type=int, default=5)
    parser.add_argument('--ranks', type=int, default=5)
    parser.add_argument('--length_bucket_num', type=int, default=5)
    parser.add_argument('--max_prompt_length', type=int, default=1000)
    parser.add_argument('--max_output_length', type=int, default=1000)

    parser.add_argument('--token_decoding_speed_coefficient1', type=float, default=2.86938135e-06)
    parser.add_argument('--token_decoding_speed_coefficient2', type=float, default=3.00834536e-02)
    parser.add_argument('--prefill_speed_coefficient1', type=float, default=1.58625005e-09)
    parser.add_argument('--prefill_speed_coefficient2', type=float, default=1.30542054e-04)
    parser.add_argument('--cache_loading_speed', type=float, default=0.00015, help='cache loading speed per token')
    parser.add_argument('--cache_saving_speed', type=float, default=0.00015, help='cache saving speed per token')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # for each time interval, we have a random number of user requests coming
    num_of_user_requests_coming = [random.randint(0, 10) for _ in range(args.gap_num)]

    # total number of user requests coming
    args.user_request_num = sum(num_of_user_requests_coming)

    config.user_request_priority, config.user_request_prompt_length, config.user_request_output_length = initialize_user_request(args)

    oracle_predicted_priority = oracle_priority_predictor(args, list(range(args.user_request_num)))
    oracle_predicted_output_bucket = oracle_output_length_bucket_predictor(args, list(range(args.user_request_num)))

    print(oracle_predicted_priority)
    print(oracle_predicted_output_bucket)
    print('---')

    simulated_predicted_priority = simulated_priority_predictor(args, list(range(args.user_request_num)), oracle_predicted_priority)
    simulated_predicted_output_bucket = simulated_output_length_bucket_predictor(args, list(range(args.user_request_num)), oracle_predicted_output_bucket)
    print(simulated_predicted_priority)
    print(simulated_predicted_output_bucket)