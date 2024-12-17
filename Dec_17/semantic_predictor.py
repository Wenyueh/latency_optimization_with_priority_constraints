import config 
import asyncio
import math
from utils import round_2

# simulated oracle predictor
def oracle_priority_predictor(user_request_ids):
    priorities = [config.user_request_priority[user_request_id] for user_request_id in user_request_ids]
    return priorities

def oracle_output_length_bucket_predictor(args,user_request_ids):
    output_length_buckets = [
        config.user_request_output_length[user_request_id] / 
        (args.max_output_length/args.length_bucket_num) 
        for user_request_id in user_request_ids
        ]
    return output_length_buckets

# helper functions
# relationship between computation time & length in prompt/output
# every token O(n) time, sum(p, p+o)
def compute_output_time(args, prompt_length, output_length):
    p = prompt_length
    o = output_length
    output_time = args.token_decoding_speed * (1/2 * o**2 + p*o + 1/2 * o)
    return round_2(output_time)
# prompt_length ^ 2 * prefill_speed
def compute_prefill_time(args, prompt_length):
    p = prompt_length
    return round_2((p ** 2)*args.prefill_speed)
# given computation length, how many tokens are computed
def compute_generated_tokens(args, prompt_length, output_time):
    a = args.token_decoding_speed/2
    b = args.token_decoding_speed * (prompt_length + 1/2) 
    c = -1 * output_time
    output_length_possibility_1 = (1/(2*a)) * (-1 * b + math.sqrt(b**2 - 4*a*c))
    output_length_possibility_2 = (1/(2*a)) * (-1 * b - math.sqrt(b**2 - 4*a*c))
    output_length = int(max(output_length_possibility_1, output_length_possibility_2))
    return output_length



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

def cache_decode_or_recompute(args, output_length):
    cache_length = output_length - math.floor(args.cache_loading_speed/args.token_decoding_speed)
    recompute_length = output_length - cache_length
    return cache_length, recompute_length
def cache_prefill_or_not(args, prompt_length):
    if args.cache_loading_speed > args.prefill_speed * prompt_length:
        return False 
    else:
        return True

# this is only called after the user request is preempted
def update_cache_loading_or_recomputation_time_and_extra_saving_time(args, user_request):
    finished_computation_time = user_request.finished_computation_time
    # next time, need to reload the cache
    user_request.prefill_cache_loaded_proportion = 0
    user_request.decoding_cache_loaded_length = 0

    prompt_length = user_request.prompt_length
    prefill_time, _, _ = user_request.predicted_remaining_computation_time
    
    # finish prefilling and now in decoding phase
    if prefill_time == 0 and user_request.should_cache_prefill:
        # finished_computation_time - compute_prefill_time(args, prompt_length) is the total time spent on decoding
        total_generated_tokens = compute_generated_tokens(args, prompt_length, finished_computation_time-compute_prefill_time(args, prompt_length))
        # compute the total number of decoded tokens that should be cached
        decoding_cache_length = min(total_generated_tokens, user_request.optimal_decoding_cache_length)
        # the number of tokens that need to be recomputed
        decoding_recompute_length = total_generated_tokens - decoding_cache_length
        # compute the time required for decoding the cache
        decoding_recompute_time = compute_output_time(args, prompt_length+decoding_cache_length, decoding_recompute_length)
        # compute the time spent on saving the new extra cache
        decoding_newly_saving_cache_length = min(decoding_cache_length-user_request.decoding_cache_length, 0)
        decoding_cache_saving_time = decoding_newly_saving_cache_length * args.cache_saving_speed
        # update decoding cache 
        user_request.decoding_cache_length = decoding_cache_length
        user_request.decoding_cache_loading_time = decoding_cache_length*args.cache_loading_speed   

        if user_request.should_cache_prefill:
            # compute the time spent on saving the new extra cache
            prefilling_newly_cache_saving_time = (1 - user_request.prefill_cache_proportion) * prompt_length * args.cache_saving_speed
            total_saving_cache_time = round_2(prefilling_newly_cache_saving_time + decoding_cache_saving_time)

            # update prefill cache
            user_request.prefill_cache_loading_time = args.cache_loading_speed * prompt_length
            user_request.prefill_cache_proportion = 1

            # update running time required
            recompute_time = decoding_recompute_time
            user_request.finished_computation_time -= recompute_time
            user_request.remaining_computation_time[1] += recompute_time
            user_request.remaining_computation_time[-1] += recompute_time
            user_request.remaining_computation_time = round_2(user_request.remaining_computation_time)
            user_request.predicted_remaining_computation_time[1] += recompute_time
            user_request.predicted_remaining_computation_time[-1] += recompute_time
            user_request.predicted_remaining_computation_time = round_2(user_request.predicted_remaining_computation_time)
        
        else:
            # update running time required
            total_saving_cache_time = round_2(decoding_cache_saving_time)
            prefilling_recompute_time = compute_prefill_time(args, prompt_length)
            user_request.prefill_cache_loading_time = 0
            user_request.prefill_cache_proportion = 0

            # total time for recomputation, both prefilling and decoding
            recompute_time = decoding_recompute_time + prefilling_recompute_time

            user_request.finished_computation_time -= recompute_time
            user_request.remaining_computation_time = [prefilling_recompute_time, user_request.remaining_computation_time[1] + decoding_recompute_time, user_request.remaining_computation_time + recompute_time]
            user_request.predicted_remaining_computation_time = [prefilling_recompute_time, user_request.predicted_remaining_computation_time[1] + decoding_recompute_time, user_request.predicted_remaining_computation_time + recompute_time] 

    # still in prefilling phase
    elif prefill_time > 0:
        if user_request.should_cache_prefill:
            newly_generated_prefill = finished_computation_time/compute_prefill_time(args, prompt_length) - user_request.prefill_cache_proportion
            # compute the time spent on saving the new extra cache
            prefilling_newly_cache_saving_time = newly_generated_prefill * prompt_length * args.cache_saving_speed
            total_saving_cache_time = round_2(prefilling_newly_cache_saving_time)

            # update prefill cache
            user_request.prefill_cache_proportion = finished_computation_time/compute_prefill_time(args, prompt_length)
            user_request.prefill_cache_loading_time = user_request.prefill_cache_proportion * prompt_length * args.cache_loading_speed

        else:
            user_request.prefill_cache_loading_time = 0
            user_request.prefill_cache_time = 0

            # if we don't cache prefill, then we need to add back the prefill time
            recompute_prefill_time = compute_prefill_time(args, prompt_length) - prefill_time
            # track back the computation time, some time will be wasted on recompute the prefill
            user_request.finished_computation_time -= recompute_prefill_time
            user_request.remaining_computation_time = [compute_prefill_time(args, prompt_length), user_request.remaining_computation_time[1], user_request.remaining_computation_time + recompute_prefill_time]
            user_request.predicted_remaining_computation_time = [compute_prefill_time(args, prompt_length), user_request.predicted_remaining_computation_time[1], user_request.predicted_remaining_computation_time + recompute_prefill_time] 

            total_saving_cache_time = 0

    # update the remaining computation time
    user_request.remaining_computation_time = round_2(user_request.remaining_computation_time)
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
