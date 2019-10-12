import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
#logging.basicConfig(level=logging.INFO)

import matplotlib.pyplot as plt


def demo_example():
    ##########################
    # From https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/
    ##########################

    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    text = "Here is the sentence I want embeddings for."
    text = "After stealing money from the bank vault, the bank robber was seen fishing on the Mississippi river bank."
    marked_text = "[CLS] " + text + " [SEP]"

    print(marked_text)

    tokenized_text = tokenizer.tokenize(marked_text)
    print(tokenized_text)

    print(list(tokenizer.vocab.keys())[5000:5020])

    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

    for tup in zip(tokenized_text, indexed_tokens):
        print(tup)

    segments_ids = [1] * len(tokenized_text)
    print(segments_ids)

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    # Load pre-trained model (weights)
    model = BertModel.from_pretrained('bert-base-uncased')

    # Put the model in "evaluation" mode, meaning feed-forward operation.
    model.eval()

    # Predict hidden states features for each layer
    with torch.no_grad():
        encoded_layers, _ = model(tokens_tensor, segments_tensors)

    print("Number of layers:", len(encoded_layers))
    layer_i = 0

    print("Number of batches:", len(encoded_layers[layer_i]))
    batch_i = 0

    print("Number of tokens:", len(encoded_layers[layer_i][batch_i]))
    token_i = 0

    print("Number of hidden units:", len(encoded_layers[layer_i][batch_i][token_i]))

    # Convert the hidden state embeddings into single token vectors

    # Holds the list of 12 layer embeddings for each token
    # Will have the shape: [# tokens, # layers, # features]
    token_embeddings = []

    # For each token in the sentence...
    for token_i in range(len(tokenized_text)):

        # Holds 12 layers of hidden states for each token
        hidden_layers = []

        # For each of the 12 layers...
        for layer_i in range(len(encoded_layers)):
            # Lookup the vector for `token_i` in `layer_i`
            vec = encoded_layers[layer_i][batch_i][token_i]

            hidden_layers.append(vec)

        token_embeddings.append(hidden_layers)
    # Sanity check the dimensions:
    print("Number of tokens in sequence:", len(token_embeddings))
    print("Number of layers per token:", len(token_embeddings[0]))

    concatenated_last_4_layers = [torch.cat((layer[-1], layer[-2], layer[-3], layer[-4]), 0) for layer in
                                  token_embeddings]  # [number_of_tokens, 3072]

    summed_last_4_layers = [torch.sum(torch.stack(layer)[-4:], 0) for layer in
                            token_embeddings]  # [number_of_tokens, 768]

    print(len(concatenated_last_4_layers))
    print(len(summed_last_4_layers))


def sen_to_bert_embedding(sentence, sum_method):
    """
    :param sentence: A sentence to be processed.
    :param sum_method:
    :return: A list of torch tensors of torch tensors that store a float.
    Each element of the list is for one of the words.
    Each of the torch tensors holds a set of torch tensors, with one float each.
    """
    # Probably don't want to set up a new one of these every time.
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    marked_text = "[CLS] " + sentence + " [SEP]"
    tokenized_text = tokenizer.tokenize(marked_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [1] * len(tokenized_text)

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    # Load pre-trained model (weights)
    model = BertModel.from_pretrained('bert-base-uncased')

    # Put the model in "evaluation" mode, meaning feed-forward operation.
    model.eval()

    # Predict hidden states features for each layer
    with torch.no_grad():
        encoded_layers, _ = model(tokens_tensor, segments_tensors)

    print("Number of layers:", len(encoded_layers))
    layer_i = 0

    print("Number of batches:", len(encoded_layers[layer_i]))
    batch_i = 0

    print("Number of tokens:", len(encoded_layers[layer_i][batch_i]))
    token_i = 0

    print("Number of hidden units:", len(encoded_layers[layer_i][batch_i][token_i]))

    token_embeddings = []

    # For each token in the sentence...
    for token_i in range(len(tokenized_text)):

        # Holds 12 layers of hidden states for each token
        hidden_layers = []

        # For each of the 12 layers...
        for layer_i in range(len(encoded_layers)):
            # Lookup the vector for `token_i` in `layer_i`
            vec = encoded_layers[layer_i][batch_i][token_i]

            hidden_layers.append(vec)

        token_embeddings.append(hidden_layers)
    # Sanity check the dimensions:
    print("Number of tokens in sequence:", len(token_embeddings))
    print("Number of layers per token:", len(token_embeddings[0]))

    if sum_method == 'concat4':
        return_vector = [torch.cat((layer[-1], layer[-2], layer[-3], layer[-4]), 0) for layer in
                                  token_embeddings]  # [number_of_tokens, 3072]
    elif sum_method == 'sum4':
        return_vector = [torch.sum(torch.stack(layer)[-4:], 0) for layer in
                            token_embeddings]  # [number_of_tokens, 768]
    else:
        print("Invalid sum method passed to sen_to_bert_embedding: ", sum_method)

    print(len(return_vector))

    return return_vector


if __name__ == '__main__':
    text = "After stealing money from the bank vault, the bank robber was seen fishing on the Mississippi river bank."
    vector = sen_to_bert_embedding(text, 'concat4')
    print(type(vector))
    print(len(vector))
    print(type(vector[0]))
    print(len(vector[0]))
    print(type(vector[0][0]))
    print(vector[0][0])
    print(type(vector[0][0].item()))
