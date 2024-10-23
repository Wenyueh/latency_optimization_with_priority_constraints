import config 


# simulated oracle predictor
def oracle_priority_predictor(user_request_id):
    return config.user_request_priority[user_request_id]

def oracle_output_length_bucket_predictor(args, user_request_id):
    return int(config.user_request_output_length[user_request_id] / (args.max_output_length/args.length_bucket_num))


# helper functions
# real output time
def compute_output_time(args, output_length):
    return output_length * args.token_decoding_speed
def compute_predicted_output_time(args, user_request_id):
    return oracle_output_length_bucket_predictor(args, user_request_id) * (args.max_output_length/args.length_bucket_num) * args.token_decoding_speed

def compute_prefill_time(args, prompt_length):
    return (prompt_length ** 2)*args.prefill_speed




def compute_total_generation_time(args, user_request_id):
    prompt_length = config.user_request_prompt_length[user_request_id]
    output_length = config.user_request_output_length[user_request_id]
    prefill_time = compute_prefill_time(args, prompt_length)
    output_time = compute_output_time(args, output_length)
    total_generation_time = prefill_time + output_time
    return prefill_time, output_time, total_generation_time
def compute_predicted_total_generation_time(args, user_request_id):
    prompt_length = config.user_request_prompt_length[user_request_id]
    prefill_time = compute_prefill_time(args, prompt_length)
    output_time = compute_predicted_output_time(args, user_request_id)
    total_generation_time = prefill_time + output_time
    return prefill_time, output_time, total_generation_time

def cache_loading_time(args, user_request_id, running_time):
    prompt_length = config.user_request_prompt_length[user_request_id]
    prefill_time, _, _ = compute_total_generation_time(args, user_request_id)
    if running_time > prefill_time:
        generated_tokens = int((running_time - prefill_time)/args.token_decoding_speed)
        return (prompt_length+generated_tokens) * args.cache_loading_speed
    else:
        return (running_time/prefill_time) * (prompt_length * args.cache_loading_speed)

def cache_saving_time(args, user_request_id, running_time):
    prompt_length = config.user_request_prompt_length[user_request_id]
    prefill_time, _, _ = compute_total_generation_time(args, user_request_id)
    if running_time > prefill_time:
        generated_tokens = int((running_time - prefill_time)/args.token_decoding_speed)
        return (prompt_length+generated_tokens) * args.cache_saving_speed
    else:
        return (running_time/prefill_time) * (prompt_length * args.cache_saving_speed)
