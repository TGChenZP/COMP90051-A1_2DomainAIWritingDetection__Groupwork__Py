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
                try:
                    new_instance['label'] = instance['label'] 
                    
                except:
                    pass
                new_instance['domain'] = instance['domain']
                new_instance['id'] = instance['id'] + (i*0.01) 
                new_instance['perplexity'] = instance['perplexity']
                new_instance['burstiness'] = instance['burstiness']
                new_instance['length'] = instance['length']
                new_instance['unique_word_ratio'] = instance['unique_word_ratio']
                
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
    raw_token_pytorch_map = {token: i+4 for i, token in enumerate(list_of_unique_train_tokens)}
    raw_token_pytorch_map['CLS'] = 0 # CLS takes on 0 in our map
    raw_token_pytorch_map['MASK'] = 1 # Masking takes 1 in our map
    raw_token_pytorch_map['PAD'] = 2 # Padding takes on 2 in our map
    raw_token_pytorch_map['LOW_FREQ'] = 3 # low_freq in our map takes value of 3
    raw_token_pytorch_map['UNK'] = 4 # Unknown takes on 4 in our map as per data (in data 1 was value of unknown)
    

    # even if we don't decide to use CLS, it doesn't affect our model at all.

    return raw_token_pytorch_map

def ReTokenise_Tokens(data, raw_token_pytorch_map, max_sentence_length, CLS=True, low_freq_special_token = False, pad_front = False):
    
    """ Convert the Token index into one useable by Pytorch Embedding Layer """
    
    data = copy.deepcopy(data)
    for instance in tqdm(data):
        instance['text'] = instance['text'] = [raw_token_pytorch_map[token] if token in raw_token_pytorch_map else raw_token_pytorch_map['LOW_FREQ' if low_freq_special_token else 'UNK'] for token in instance['text']]
        if CLS: 
            instance['text'] = [raw_token_pytorch_map['CLS']] + instance['text']

    if max_sentence_length:
        for instance in data:
            if len(instance['text']) < max_sentence_length:
                if pad_front:
                    instance['text'] = [raw_token_pytorch_map['PAD']] * (max_sentence_length - len(instance['text'])) + instance['text']
                else:
                    instance['text'] = instance['text'] + [raw_token_pytorch_map['PAD']] * (max_sentence_length - len(instance['text'])) # 1 is pad in our map
            else:
                instance['text'] = instance['text'][:max_sentence_length]

    return data

def Data_Factory(train_data, val_data, test_data, future_data, max_sentence_length, raw_token_pytorch_map, CLS=True, low_freq_special_token=False, pad_front=False):
    
    """ Convert our (cropped) data into train x, train y, val x, val y, test x, test y etc """

    train_data_transformed = ReTokenise_Tokens(train_data, raw_token_pytorch_map, max_sentence_length, CLS, low_freq_special_token, pad_front)
    val_data_transformed = ReTokenise_Tokens(val_data, raw_token_pytorch_map, max_sentence_length, CLS, low_freq_special_token, pad_front)
    test_data_transformed = ReTokenise_Tokens(test_data, raw_token_pytorch_map, max_sentence_length, CLS, low_freq_special_token, pad_front)
    future_data_transformed = ReTokenise_Tokens(future_data, raw_token_pytorch_map, max_sentence_length, CLS, low_freq_special_token, pad_front)

    train_x = [instance['text'] for instance in train_data_transformed]
    train_y = [instance['label'] for instance in train_data_transformed]
    val_x = [instance['text'] for instance in val_data_transformed]
    val_y = [instance['label'] for instance in val_data_transformed]
    test_x = [instance['text'] for instance in test_data_transformed]
    test_y = [instance['label'] for instance in test_data_transformed]
    train_dom = [instance['domain'] for instance in train_data_transformed]
    val_dom = [instance['domain'] for instance in val_data_transformed]
    test_dom = [instance['domain'] for instance in test_data_transformed]

    future_x = [instance['text'] for instance in future_data_transformed]
    future_dom = [instance['domain'] for instance in future_data_transformed]

    train_y = [[0, 1] if label == 1 else [1, 0] for label in train_y] 
    val_y = [[0, 1] if label == 1 else [1, 0] for label in val_y]
    test_y = [[0, 1] if label == 1 else [1, 0] for label in test_y]

    train_dom = [[0, 1] if domain == 1 else [1, 0] for domain in train_dom]
    val_dom = [[0, 1] if domain == 1 else [1, 0] for domain in val_dom]
    test_dom = [[0, 1] if domain == 1 else [1, 0] for domain in test_dom]

    return train_x, train_y, val_x, val_y, test_x, test_y, train_dom, val_dom, test_dom, future_x, future_dom

def Data_Factory_aux(train_data, val_data, test_data, future_data, features):
    
    """ Convert our (cropped) data into train x, train y, val x, val y, test x, test y etc """

    train_aux = [[instance[feature] for feature in features] for instance in train_data]
    val_aux = [[instance[feature] for feature in features] for instance in val_data]
    test_aux = [[instance[feature] for feature in features] for instance in test_data]
    future_aux = [[instance[feature] for feature in features] for instance in future_data]

    return train_aux, val_aux, test_aux, future_aux
    
def get_distribution(train_y) -> Tuple[float, float]:

    """ get the distribution of labels in this set - for processing the loss function """

    label = [y[1] for y in train_y]

    return np.mean(label), 1-np.mean(label)

def W2V_DataFactory(data: list, context_window: int, seed: int, raw_token_pytorch_map: dict, k) -> list:

    """ Get W2V training data """
    
    assert context_window % 2 == 1, 'context window must be odd'

    np.random.seed(seed)

    MAX_SAMPLED_NEGATIVE_TOKENS = 10000

    retokenised_keys = list(raw_token_pytorch_map.keys())

    negative_tokens = np.random.choice(retokenised_keys, MAX_SAMPLED_NEGATIVE_TOKENS)
    negative_tokens = [x if x[0].isalpha() else int(x) for x in negative_tokens]

    negative_up_to = 0

    w2v_data = []

    for instance in tqdm(data): # every sentence
        tokens = [context_window//2 * 'CLS'] + instance['text'] + [context_window//2 * raw_token_pytorch_map['PAD']]

        for i in range(context_window//2, len(tokens) - context_window//2): # avoid pad and cls # every token
            
            focus_token_retokenised = raw_token_pytorch_map.get(tokens[i], raw_token_pytorch_map['UNK'])
            
            context_words = dict()

            for j in range(-context_window//2+1, context_window//2+1):
                if j != 0:
                    context_words[tokens[i+j]] = 0

            for j in range(-context_window//2+1, context_window//2+1): # every neighbour in window
                if j != 0: # don't want to make positive sample with self
                    if context_words.get(tokens[i+j], 0): # CLS and Padding (being start and end) being repeated
                        continue 
                    else:
                        context_words[tokens[i+j]] = 1
                    
                    mask = [raw_token_pytorch_map.get(tokens[i+j], raw_token_pytorch_map['UNK'])]
                    for _ in range(k): # sample the same number of negatives
                        
                        while True:
                            
                            if negative_up_to == MAX_SAMPLED_NEGATIVE_TOKENS:
                                negative_up_to = 0

                            sampled_negative_retokenised = negative_tokens[negative_up_to]
                            negative_up_to += 1

                            if sampled_negative_retokenised not in context_words: # didn't sample a positive case
                                mask.append(raw_token_pytorch_map[sampled_negative_retokenised])
                                break

                    
                    new_instance = {'token': focus_token_retokenised, 'mask': mask}
                    w2v_data.append(new_instance)
    
    return w2v_data


def BERT_pretrain_Generation(data: list, seed: int, raw_token_pytorch_map: dict, MAX_SENTENCE_LENGTH):
    
    np.random.seed(seed)

    MAX_SAMPLED_PROBS = 10000

    mask_randomness = np.random.uniform(0, 1, size=MAX_SAMPLED_PROBS)

    negative_up_to = 0

    bert_data = []

    for instance in tqdm(data):
        

        tokens = instance['text']

        tokens = [raw_token_pytorch_map['CLS']] + [raw_token_pytorch_map.get(token, raw_token_pytorch_map['UNK']) for token in tokens]
        tokens = tokens[:MAX_SENTENCE_LENGTH]

        # 15% of tokens are random
        masked_token_positions = np.random.choice(range(len(tokens)), int(0.15 * len(tokens)), False)
        
        # 80% becomes [MASK], 15% becomes random, of which 10% unchanged ith token, 10% random token
        for masked_token_position in masked_token_positions:
            if mask_randomness[negative_up_to] < 0.8:
                tokens[masked_token_position] = raw_token_pytorch_map['MASK']
            elif mask_randomness[negative_up_to] > 0.9:
                tokens[masked_token_position] = np.random.choice(tokens)
            
        
        tokens = tokens + [raw_token_pytorch_map['PAD']] * (MAX_SENTENCE_LENGTH - len(tokens))

        for masked_token_position in masked_token_positions:
            new_instance = {}
            new_instance['input'] = tokens
            new_instance['label'] = tokens[masked_token_position]
            new_instance['mask'] = masked_token_position
            new_instance['domain'] = instance['domain']
            bert_data.append(new_instance)
    
    return bert_data

def BERT_pretrain_DataFactory(train_data, val_data, seed, raw_token_pytorch_map, MAX_SENTENCE_LENGTH):
        
    train_data = BERT_pretrain_Generation(train_data, seed, raw_token_pytorch_map, MAX_SENTENCE_LENGTH)
    val_data = BERT_pretrain_Generation(val_data, seed, raw_token_pytorch_map, MAX_SENTENCE_LENGTH)

    train_x = [instance['input'] for instance in train_data]
    train_y = [instance['label'] for instance in train_data]
    train_mask = [instance['mask'] for instance in train_data]
    train_domain = [instance['domain'] for instance in train_data]
    val_x = [instance['input'] for instance in val_data]
    val_y = [instance['label'] for instance in val_data]
    val_mask = [instance['mask'] for instance in val_data]
    val_domain = [instance['domain'] for instance in val_data]
    

    return train_x, train_y, train_mask, train_domain, val_x, val_y, val_mask, val_domain