import sys
import os
import math
import random
import yaml
import argparse
import pickle
import queue as Q

import numpy as np
import tensorflow as tf

# from tensorflow.keras import backend
# from keras_self_attention import SeqSelfAttention

random.seed(41)
from data_reader import DataReader

config = yaml.safe_load(open("config.yml"))

log_2_e = 1.44269504089  # Constant to convert to binary entropies


class MyFloat(float):
    def __lt__(self, other):
        return self > other


class WordModel(tf.keras.Model):
    def __init__(self, embed_dim, hidden_dim, num_layers, vocab_dim):
        super(WordModel, self).__init__()
        random_init = tf.random_normal_initializer(stddev=0.1)
        self.embed = tf.Variable(random_init([vocab_dim, embed_dim]), dtype=tf.float32)
        self.rnns = [tf.keras.layers.GRU(hidden_dim, return_sequences=True) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(0.5)
        # self.attention = SeqSelfAttention(attention_activation='sigmoid')
        self.project = tf.keras.layers.Dense(vocab_dim)

    # Very basic RNN-based language model: embed inputs, encode through several layers and project back to vocabulary
    def call(self, indices, training=True):
        states = tf.nn.embedding_lookup(self.embed, indices)
        for ix, rnn in enumerate(self.rnns):
            states = rnn(states)
        # states = self.attention(states)
        states = self.dropout(states)
        preds = self.project(states)
        return preds


def eval(model, data):
    mbs = 0
    count = 0
    entropy = 0.0
    for indices, masks in data.batcher(data.valid_data, is_training=False):
        mbs += 1
        samples = int(tf.reduce_sum(masks[:, 1:]).numpy())
        count += samples
        preds = model(indices[:, :-1])
        loss = masked_ce_loss(indices, masks, preds)
        entropy += log_2_e * float(samples * loss.numpy())
    entropy = entropy / count
    return entropy, count


# Compute cross-entropy loss, making sure not to include "masked" padding tokens
def masked_ce_loss(indices, masks, preds):
    samples = tf.reduce_sum(masks[:, 1:])
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(indices[:, 1:], preds.shape[-1]), logits=preds)
    loss *= masks[:, 1:]
    loss = tf.reduce_sum(loss) / samples
    return loss


def train(model, data):
    # Declare the learning rate as a variable to include it in the saved state
    learning_rate = tf.Variable(config["training"]["lr"], name="learning_rate")
    optimizer = tf.keras.optimizers.Adam(learning_rate)

    is_first = True
    for epoch in range(config["training"]["num_epochs"]):
        print("Epoch:", epoch + 1)
        mbs = 0
        words = 0
        avg_loss = 0
        # Batcher returns a square index array and a binary mask indicating which words are padding (0) and real (1)
        for indices, masks in data.batcher(data.train_data):
            mbs += 1
            samples = tf.reduce_sum(masks[:, 1:])
            words += int(samples.numpy())

            # Run through one batch to init variables
            if is_first:
                model(indices[:, :-1])
                is_first = False

            # Compute loss in scope of gradient-tape (can also use implicit gradients)
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(model.variables)
                preds = model(indices[:, :-1])
                loss = masked_ce_loss(indices, masks, preds)

            # Collect gradients, clip and apply
            grads = tape.gradient(loss, model.variables)
            grads, _ = tf.clip_by_global_norm(grads, 0.25)

            optimizer.apply_gradients(zip(grads, model.variables))
            # Update average loss and print if applicable
            avg_loss += log_2_e * loss
            if mbs % config["training"]["print_freq"] == 0:
                avg_loss = avg_loss.numpy() / config["training"]["print_freq"]
                print("MB: {0}: words: {1}, entropy: {2:.3f}".format(mbs, words, avg_loss))
                avg_loss = 0.0

        # Run a validation pass at the end of every epoch
        entropy, count = eval(model, data)
        print("Validation: tokens: {0}, entropy: {1:.3f}, perplexity: {2:.3f}".format(count, entropy,
                                                                                      0.0 if entropy > 100 else math.pow(
                                                                                          2, entropy)))


def index_to_word(index, data):
    word = data.i2w[index.numpy()]
    return word


def indices_to_word(indices, data):
    words = []
    for index in indices:
        words.append(data.i2w[index.numpy()])
    print("!!!!!!!!!!!Print word given indices!!!!!!!!!!")
    print(words)
    return words


def indices_to_word2(indices, data):
    words = []
    for index in indices:
        words.append(data.i2w[index])
    print("!!!!!!!!!!!Print word given indices!!!!!!!!!!")
    print(words)
    return words


# find the top K prediction form preds, which is generated by model(input_indices)
def generate_K_indices_pq(preds, input_masks, data, K):
    listA = preds.shape.as_list()
    indices_result = np.zeros((listA[0] * K, listA[1]))
    logits = np.zeros(K)
    probabilities = np.zeros(K)
    index_to_expand = tf.reduce_sum(input_masks[0, :])

    for i in range((listA[0])):
        # directly go to to the last word of prediction as this is the only word we care about
        for j in range(index_to_expand - 1, index_to_expand):
            # get probability from logits
            prediction = tf.nn.softmax(preds[i, j, :])

            (logits, topK_logits_index) = tf.math.top_k(preds[i, j, :], K)
            for k in range(K):
                indices_result[i + k * listA[0], j] = topK_logits_index[k]
                probabilities[k] = prediction[topK_logits_index[k]]

    return tf.constant(indices_result, dtype="int32"), logits, probabilities, index_to_expand


def beam_search_pq(model, data, input_indices_tf, input_mask_tf, K):
    suffix = "#"
    min_heap_result = Q.PriorityQueue()
    max_heap_candidate = Q.PriorityQueue()
    # add dummy threshold node
    min_heap_result.put((0.00001, input_indices_tf, input_mask_tf))
    # unique count
    count = 0

    # initialize the max_heap_candidate by pushing the first K top prediction of input indices
    index_tf = input_indices_tf[:-1]
    expand_index_tf = tf.expand_dims(index_tf, 0)
    expand_mask_tf = tf.expand_dims(input_mask_tf, 0)
    preds = model(expand_index_tf)
    (k_indices_tf, logits, probabilities, index_to_expand) = generate_K_indices_pq(preds, expand_mask_tf, data, K)
    for i in range(K):
        new_index_tf = tf.Variable(input_indices_tf)
        new_index_tf = new_index_tf[int(index_to_expand.numpy())].assign(k_indices_tf[i, int(index_to_expand.numpy() - 1)])
        new_mask_tf = tf.Variable(input_mask_tf)
        new_mask_tf = new_mask_tf[int(index_to_expand.numpy())].assign(tf.constant(1.0, dtype="float32"))
        # add unique count
        count = count + 1
        # push minus probability as we want a max heap
        max_heap_candidate.put((-probabilities[i], count, (index_to_expand, new_index_tf, new_mask_tf)))


    while not max_heap_candidate.empty():
        max_heap_candidate_top = max_heap_candidate.get()
        probability = -max_heap_candidate_top[0]
        indices = max_heap_candidate_top[2][1]
        mask = max_heap_candidate_top[2][2]
        index_to_expand = max_heap_candidate_top[2][0]

        # if cur word is not an end of token
        if (index_to_word(indices[int(index_to_expand.numpy())], data)).endswith(suffix):
            listA = indices.shape.as_list()
            # if already reach end, i.e. no more space for additional search,
            # directly push the not end token to result pq
            if int(index_to_expand.numpy()) == listA[0] - 1:
                min_heap_result.put((probability, indices, mask))
                # keep result size less than or equal to K
                if min_heap_result.qsize() > K:
                    min_heap_result.get()
            # keep searching
            else:
                min_heap_result_top = min_heap_result.get()
                min_heap_result.put(min_heap_result_top)
                # only search on beams that have higher probability than a threshold,
                # which is the min probability of cur results
                if min_heap_result_top[0] < probability:
                    index_tf = indices[:-1]
                    input_mask_tf = mask
                    expand_index_tf = tf.expand_dims(index_tf, 0)
                    expand_mask_tf = tf.expand_dims(input_mask_tf, 0)
                    preds = model(expand_index_tf)
                    (k_indices_tf, logits, probabilities, index_to_expand) = generate_K_indices_pq(preds,
                                                                                                       expand_mask_tf,
                                                                                                       data, K)
                    for i in range(K):
                        new_index_tf = tf.Variable(indices)
                        new_index_tf = new_index_tf[int(index_to_expand.numpy())].assign(
                            k_indices_tf[i, int(index_to_expand.numpy() - 1)])
                        new_mask_tf = tf.Variable(input_mask_tf)
                        new_mask_tf = new_mask_tf[int(index_to_expand.numpy())].assign(
                            tf.constant(1.0, dtype="float32"))
                        # add unique count
                        count = count + 1
                        max_heap_candidate.put(
                            (-(probability * probabilities[i]), count, (index_to_expand, new_index_tf, new_mask_tf)))
                else:
                    break
        else:
            min_heap_result.put((probability, indices, mask))
            # keep result size less than or equal to K
            if min_heap_result.qsize() > K:
                min_heap_result.get()
    return min_heap_result


# generate list of size of word sequence and mask by using validation batch data
def generate_indices_masks(data, size, random_seed):
    masks_result = []
    indices_result = []

    # pass in a random seed to shuffle data so that get same data every time
    random.Random(random_seed).shuffle(data.valid_data)

    # IMPORTANT: grab the target number of word sequence from the first batch of validation batch data,
    # will cause error if the input word sequence size is larger than the actual first validation batch size
    # the reason is that I try to reuse the code as much as possible
    for indices, masks in data.batcher(data.valid_data, is_training=False):
        indices_result.extend(indices[:size, :])
        masks_result.extend(masks[:size, :])
        break
    return indices_result, masks_result


# generate measure positions for MRR in a given word sequence
def generate_measure_positions(data, indices, mask):
    indices_result = []
    masks_result = []
    cur_indices = []
    cur_mask = []

    for i in range(201):
        if mask[i].numpy() == 0:
            break

        # every time iterate to a new word, add it to the end of cur_indices
        cur_indices.append(int(indices[i].numpy()))
        cur_mask.append(float(mask[i].numpy()))

        # if cur word is an end of token, i.e. not ending with '#', append the cur_indices to result list
        if not data.i2w[int(indices[i].numpy())].endswith("#"):
            indices_result.append(cur_indices.copy())
            masks_result.append(cur_mask.copy())

    return indices_result, masks_result


# get the MRR score for a priority queue of beam search reseult from a measuring position,
def get_score(pq, input_indices_tf_next, input_mask_tf_next, K, data):
    base = K + 1
    while not pq.empty():
        base = base - 1
        [probability, indices, masks] = pq.get()
        listA = input_indices_tf_next.shape.as_list()
        same = True
        for i in range(listA[0]):
            # if find a mismatch, break the inner loop
            if not int(indices[i].numpy()) == int(input_indices_tf_next[i].numpy()):
                same = False
                break
        # if find one exactly same, direectly return
        if same:
            return 1 / base
    return 0


def measure_MRR(model, data, beam_search_size_K, sequences_to_measure, random_seed):
    K = beam_search_size_K

    measure_count = 0
    total_measure_score = 0

    # generate a list of word index sequence and a list of correspondent mask with size(sequences_to_measure)
    [indices_list, masks_list] = generate_indices_masks(data, sequences_to_measure, random_seed)

    for i in range(sequences_to_measure):
        masks_i = masks_list[i]
        indices_i = indices_list[i]
        # generate measure positions from a sequence
        [indices, masks] = generate_measure_positions(data, indices_i, masks_i)

        # remain some space at head and tail, mainly to avoid out of boundary error
        for j in range(3, len(indices) - 5):
            cur_indices = indices[j].copy()
            cur_masks = masks[j].copy()
            measure_count = measure_count + 1

            # patching zero with target token size (in word) at the end of input_indices to beam search,
            # add 5 extra because we need some overhead for beam search;
            # however, any prediction use the extra 5 word space will not match target token as they are longer
            zero_to_add = len(indices[j + 1]) - len(indices[j]) + 5
            cur_indices.extend([0] * zero_to_add)
            cur_masks.extend([0.0] * zero_to_add)

            input_indices_tf = tf.constant(cur_indices, dtype="int32")
            input_mask_tf = tf.constant(cur_masks, dtype="float32")
            pq = beam_search_pq(model, data, input_indices_tf, input_mask_tf, K)
            total_measure_score = total_measure_score + get_score(pq, tf.constant(indices[j + 1], dtype="int32"),
                                                tf.constant(masks[j + 1], dtype="float32"), K, data)
        MRR = total_measure_score / measure_count
        print("currently measure sequence:")
        print(i)
        print("accumulative MRR:")
        print(MRR)
    MRR = total_measure_score / measure_count
    return MRR


def main():
    # Extract arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("train_data", help="Path to training data")
    ap.add_argument("-v", "--valid_data", required=False, help="(optional) Path to held-out (validation) data")
    ap.add_argument("-t", "--test_data", required=False, help="(optional) Path to test data")
    args = ap.parse_args()
    print("Using configuration:", config)
    data = DataReader(args.train_data, args.valid_data, args.test_data)
    model = WordModel(config["model"]["embed_dim"], config["model"]["hidden_dim"], config["model"]["num_layers"],
                      data.vocab_dim)


    ########## SAVE TRAINED MODEL AND DATA ##########
    # random.shuffle(data.valid_data)  # Shuffle just once
    # train(model, data)
    # model.save_weights('test_model_2')
    # f = open('store_2.pckl', 'wb')
    # pickle.dump(data, f)
    # f.close()

    ########## LOAD TRAINED MODEL AND DATA ##########
    # the currently loaded data following is a 10 epoch trained data
    f = open('store_1.pckl', 'rb')
    data = pickle.load(f)
    f.close()
    model.load_weights(os.path.join(os.sep, "home", "program", "code", "test_model_1"))

    beam_search_size_K = 10
    sequences_to_measure = 1
    random_seed = 1032
    measure_MRR(model, data, beam_search_size_K, sequences_to_measure, random_seed)

    print("Everything Done")


if __name__ == '__main__':
    main()
