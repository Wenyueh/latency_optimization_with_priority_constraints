import json, argparse, time, random, os
from tqdm import tqdm   
import numpy as np

def compute_chat_length_distribution(args, splitted_chats):
    # classify the length of the chat into 10 bins, and then show the distribution of the length of the chat
    # use the min and max length of the chat to determine the range of each bin
    bins = {}
    min_len = np.min([len(chat[1]['content'].split()) for chat in splitted_chats])
    max_len = 50
    real_max_len = np.max([len(chat[1]['content'].split()) for chat in splitted_chats])
    for i in range(args.number_of_bins-1):
        ran = (min_len + int((max_len - min_len) / args.number_of_bins * i), min_len + int((max_len - min_len) / args.number_of_bins * (i + 1)))
        bins[ran] = []
    bins[(min_len + int((max_len - min_len) / args.number_of_bins * (args.number_of_bins-1)), real_max_len)] = []

    orders = list(bins.keys())
    for chat_id, chat in enumerate(splitted_chats):
        length = len(chat[1]['content'].split())
        for ran in bins:
            if ran[0] <= length < ran[1]:
                bins[ran].append(chat[1]['content'])
                order = orders.index(ran)
                splitted_chats[chat_id][-1] = [splitted_chats[chat_id][-1], order]
                break
    
    print(bins.keys())
    print([len(bins[ran]) for ran in bins])
    print(splitted_chats[0])
    # if 10 keys: dict_keys([(3, 7), (7, 11), (11, 15), (15, 19), (19, 24), (24, 28), (28, 32), (32, 36), (36, 40), (40, 63)])
    # if 6 keys: dict_keys([(3, 10), (10, 17), (17, 24), (24, 31), (31, 38), (38, 63)])
    # for chats in bins[(36,40)]:
    #     print(chats)
    #     print()
    with open(f'data/splitted_chats_with_{args.number_of_bins}_bins.json', 'w') as f:
        json.dump(splitted_chats, f)

def compute_shared_prefix(splitted_chats):
    # compute average shared prefix for each chat
    user_prompt = [d[0]['content'].split() for d in splitted_chats][:100]
    # for each chat, compute the average shared prefix with all other chats
    shared_prefix = []
    for i in tqdm(range(len(user_prompt))):
        shared_prefix.append([])
        for j in range(len(user_prompt)):
            if i == j:
                continue
            else:
                shared_prefix[-1].append(len(os.path.commonprefix([user_prompt[i], user_prompt[j]])))

    for i in range(len(shared_prefix)):
        shared_prefix[i] = np.max(shared_prefix[i])

    print(np.mean(shared_prefix))

# load the data and split the chat into user and assistant
def load_original_data(args):
    with open(args.data) as f:
        data = json.load(f)

    splitted_chats = []
    for chat in tqdm(data):
        pair = []
        for one_chat in chat[1:]:
            if one_chat['role'] == 'user':
                pair.append(one_chat)
            if one_chat['role'] == 'assistant':
                if 'tool_calls' not in one_chat:
                    pair.append(one_chat)
                    splitted_chats.append(pair)
                    pair = []

    with open('data/splitted_chats.json', 'w') as f:
        json.dump(splitted_chats, f)

    return splitted_chats

def load_original_data_for_semantics(args):
    with open(args.data) as f:
        data = json.load(f)

    splitted_chats = []
    for chat in tqdm(data):
        pair = []
        number_of_rounds = int((len(chat)-1)/2)
        for one_chat_id, one_chat in enumerate(chat[1:]):
            if one_chat['role'] == 'user':
                pair.append(one_chat)
            if one_chat['role'] == 'assistant':
                if 'tool_calls' not in one_chat:
                    pair.append(one_chat)
                    if one_chat_id <= number_of_rounds/5:
                        if 'bleeding' in pair[0]['content'] or 'blood' in pair[0]['content'] or 'heart' in pair[0]['content']:
                            pair.append(1)
                        else:
                            pair.append(random.choice([1, 2]))
                    elif one_chat_id <= number_of_rounds/5*2 and one_chat_id > number_of_rounds/5:
                        if 'bleeding' in pair[0]['content'] or 'blood' in pair[0]['content'] or 'heart' in pair[0]['content']:
                            pair.append(1)
                        else:
                            pair.append(random.choice([2, 3]))
                    elif one_chat_id <= number_of_rounds/5*3 and one_chat_id > number_of_rounds/5*2:
                        if 'bleeding' in pair[0]['content'] or 'blood' in pair[0]['content'] or 'heart' in pair[0]['content']:
                            pair.append(1)
                        else:
                            pair.append(random.choice([3, 4]))
                    elif one_chat_id <= number_of_rounds/5*4 and one_chat_id > number_of_rounds/5*3:
                        if 'bleeding' in pair[0]['content'] or 'blood' in pair[0]['content'] or 'heart' in pair[0]['content']:
                            pair.append(1)
                        else:
                            pair.append(random.choice([4, 5]))
                    elif one_chat_id <= number_of_rounds/5*5 and one_chat_id > number_of_rounds/5*4:
                        if 'bleeding' in pair[0]['content'] or 'blood' in pair[0]['content'] or 'heart' in pair[0]['content']:
                            pair.append(1)
                        else:
                            pair.append(5)
                    else:
                        pair.append(5)
                    splitted_chats.append(pair)
                    pair = []

    with open('data/splitted_chats_with_semantic_label.json', 'w') as f:
        json.dump(splitted_chats, f)

    return splitted_chats

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/chat_histories.json')
    parser.add_argument('--number_of_bins', type=int, default=10)
    parser.add_argument('--number', type=int, default=0)
    args = parser.parse_args()

    load_original_data(args)
    splitted_chats = load_original_data_for_semantics(args)
    compute_chat_length_distribution(args, splitted_chats)


