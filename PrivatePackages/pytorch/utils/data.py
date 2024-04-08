from environment import *

def crop_sentence_length(data: list, max_sentence_length: int, make_cropped_remains_into_new_instance: bool) -> dict:
    
    """ For every sentence, reduce the length of sentence to a fixed value 
    if it is over the max_length """

    new_data = []

    for instance in tqdm(data):
        for i in range(len(data)//max_sentence_length+1):

            new_instance = {}
            
            cropped_text = instance['text'][i*max_sentence_length:(i+1)*max_sentence_length]

            if len(cropped_text) == 0:
                continue

            if i == 0 or make_cropped_remains_into_new_instance:

                new_instance['text'] = cropped_text
                new_instance['label'] = instance['label'] 
                new_instance['domain'] = instance['domain']
                new_instance['id'] = instance['id'] + (i*0.01) # TODO
                
                new_instance['remains'] = 0 if i == 0 else 1 
                
                new_data.append(new_instance)

            else:
                break

    return new_data


def get_raw_token_pytorch_map(data: list, min_frequency: int = 0) -> dict:

    """ get a mapping that:
        1. places restrictions on how many times an instance is observed to be eligible to be learnt in model as an embedding
        2. get a map of raw token values to compressed and sorted token values, with addition of CLS, Padding and Unknown.
    """
    
    tally_of_unique_train_tokens = dd(int)
    for instance in tqdm(data):
        for token in instance['text']:
            tally_of_unique_train_tokens[token] += 1
    
    if 1 not in tally_of_unique_train_tokens: # ensure that 1 is in there so not to mess up our mapping (1 is this dataset's unknown)
        tally_of_unique_train_tokens[1] = np.inf

    list_of_unique_train_tokens = list(tally_of_unique_train_tokens.items())
    list_of_unique_train_tokens = [token_count[0] for token_count in list_of_unique_train_tokens if token_count[1] > min_frequency]
    list_of_unique_train_tokens.sort() # so to keep our original tokens in order

    # Need to compress the used tokens (in training) onto a denser map for pytorch embedding
    raw_token_pytorch_map = {token: i+2 for i, token in enumerate(list_of_unique_train_tokens)}
    raw_token_pytorch_map['CLS'] = 0 # CLS takes on 0 in our map
    raw_token_pytorch_map['PAD'] = 1 # Padding takes on 1 in our map
    raw_token_pytorch_map['UNK'] = 2 # Unknown takes on 2 in our map as per data (in data 1 was value of unknown)

    # even if we don't decide to use CLS, it doesn't affect our model at all.

    return raw_token_pytorch_map

def ReTokenise_Tokens(data, raw_token_pytorch_map, max_sentence_length, CLS=True):
    
    """ Convert the Token index into one useable by Pytorch Embedding Layer """
    
    data = copy.deepcopy(data)
    for instance in tqdm(data):
        instance['text'] = instance['text'] = [raw_token_pytorch_map[token] if token in raw_token_pytorch_map else raw_token_pytorch_map['UNK'] for token in instance['text']]
        if CLS: 
            instance['text'] = [raw_token_pytorch_map['CLS']] + instance['text']

    if max_sentence_length:
        for instance in data:
            if len(instance['text']) < max_sentence_length:
                instance['text'] = instance['text'] + [raw_token_pytorch_map['PAD']] * (max_sentence_length - len(instance['text'])) # 1 is pad in our map
            else:
                instance['text'] = instance['text'][:max_sentence_length]

    return data

def Data_Factory(train_data, val_data, test_data, max_sentence_length, raw_token_pytorch_map, CLS=True):
    
    """ Convert our (cropped) data into train x, train y, val x, val y, test x, test y etc """

    train_data_transformed = ReTokenise_Tokens(train_data, raw_token_pytorch_map, max_sentence_length, CLS)
    val_data_transformed = ReTokenise_Tokens(val_data, raw_token_pytorch_map, max_sentence_length, CLS)
    test_data_transformed = ReTokenise_Tokens(test_data, raw_token_pytorch_map, max_sentence_length, CLS)

    train_x = [instance['text'] for instance in train_data_transformed]
    train_y = [instance['label'] for instance in train_data_transformed]
    val_x = [instance['text'] for instance in val_data_transformed]
    val_y = [instance['label'] for instance in val_data_transformed]
    test_x = [instance['text'] for instance in test_data_transformed]
    test_y = [instance['label'] for instance in test_data_transformed]
    train_dom = [instance['domain'] for instance in train_data_transformed]
    val_dom = [instance['domain'] for instance in val_data_transformed]
    test_dom = [instance['domain'] for instance in test_data_transformed]

    train_y = [[0, 1] if label == 1 else [1, 0] for label in train_y] #TODO: check this
    val_y = [[0, 1] if label == 1 else [1, 0] for label in val_y]
    test_y = [[0, 1] if label == 1 else [1, 0] for label in test_y]

    train_dom = [[0, 1] if domain == 1 else [1, 0] for domain in train_dom]
    val_dom = [[0, 1] if domain == 1 else [1, 0] for domain in val_dom]
    test_dom = [[0, 1] if domain == 1 else [1, 0] for domain in test_dom]

    return train_x, train_y, val_x, val_y, test_x, test_y, train_dom, val_dom, test_dom
    
def get_distribution(train_y) -> Tuple[float, float]:

    """ get the distribution of labels in this set - for processing the loss function """

    label = [y[1] for y in train_y]

    return np.mean(label), 1-np.mean(label)