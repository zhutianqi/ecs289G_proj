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


def generate_K_indices(preds, input_masks, data, K, input_logits):
    listA = preds.shape.as_list()
    indices_result = np.zeros((listA[0] * K, listA[1]))
    logits = np.zeros(K)
    probability = np.zeros(K)
    for i in range((listA[0])):
        index_to_expand = tf.reduce_sum(input_masks[i, :])
        print("index_to_expand = ")
        print(index_to_expand)
        for j in range(index_to_expand - 1):
            print("j =")
            print(j)
            (values, topK_logits_index) = tf.math.top_k(preds[i, j, :], 1)
            print(preds[i, j, :])
            print(topK_logits_index)
            for k in range(K):
                indices_result[i + k * listA[0], j] = topK_logits_index[0] + 1
        for j in range(index_to_expand - 1, index_to_expand):
            print("j =")
            print(j)
            (values, topK_logits_index) = tf.math.top_k(preds[i, j, :], K)
            logits = values + input_logits
            print("logits:")
            print(logits)
            print(preds[i, j, :])
            print(topK_logits_index)
            for k in range(K):
                indices_result[i + k * listA[0], j] = topK_logits_index[k] + 1
    for i in range(k):
        probability[i] = logit_to_probability(logits[i])
    return tf.constant(indices_result, dtype="int32"), tf.constant(logits, dtype="float32")


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


def logit_to_probability(logit):
    odds = math.exp(logit)
    probability = odds / (1 + odds)
    return probability


def generate_K_indices_pq(preds, input_masks, data, K):
    listA = preds.shape.as_list()
    indices_result = np.zeros((listA[0] * K, listA[1]))
    logits = np.zeros(K)
    probabilities = np.zeros(K)
    index_to_expand = tf.reduce_sum(input_masks[0, :])
    for i in range((listA[0])):
        # for j in range(index_to_expand - 1):
        #     (values, topK_logits_index) = tf.math.top_k(preds[i, j, :], 1)
        #     for k in range(K):
        #         indices_result[i + k * listA[0], j] = topK_logits_index[0] + 1
        for j in range(index_to_expand - 1, index_to_expand):
            prediction = tf.nn.softmax(preds[i, j, :])
            (logits, topK_logits_index) = tf.math.top_k(preds[i, j, :], K)
            for k in range(K):
                # do we need to add 1? ie how to deal with index 0
                indices_result[i + k * listA[0], j] = topK_logits_index[k]
                probabilities[k] = prediction[topK_logits_index[k]]
    # print("probabilities:")
    # print(probabilities)
    return tf.constant(indices_result, dtype="int32"), logits, probabilities, index_to_expand


def beam_search_pq(model, data, input_indices_tf, mask_tf, K):
    min_heap_result = Q.PriorityQueue()
    max_heap_candidate = Q.PriorityQueue()
    # dummy
    min_heap_result.put((0.00001, input_indices_tf, mask_tf))
    #unique count
    count = 0

    # initialize the max_heap_candidate
    index_tf = input_indices_tf[:-1]
    expand_index_tf = tf.expand_dims(index_tf, 0)
    expand_mask_tf = tf.expand_dims(mask_tf, 0)
    preds = model(expand_index_tf)
    (k_indices_tf, logits, probabilities, index_to_expand) = generate_K_indices_pq(preds, expand_mask_tf, data, K)
    # print("!!!!!!!!!!")
    # print(logits)
    # prediction = tf.nn.softmax(logits)
    # print(prediction)
    # print(probabilities)
    # print("!!!!!!!!!!")
    for i in range(K):
        new_index_tf = tf.Variable(input_indices_tf)
        new_index_tf = new_index_tf[int(index_to_expand.numpy())].assign(
            k_indices_tf[i, int(index_to_expand.numpy() - 1)])
        new_mask_tf = tf.Variable(mask_tf)
        new_mask_tf = new_mask_tf[int(index_to_expand.numpy())].assign(tf.constant(1.0, dtype="float32"))
        # print(input_indices_tf)
        # print(mask_tf)
        # print(new_index_tf)
        # print(new_mask_tf)
        # if (probabilities[i] > 0.0001):
        count = count + 1
        max_heap_candidate.put((-probabilities[i], count, (index_to_expand, new_index_tf, new_mask_tf)))
        # max_heap_candidate.put((-logits[i], index_to_expand, new_index_tf, new_mask_tf))

    while not max_heap_candidate.empty():
        max_heap_candidate_top = max_heap_candidate.get()
        # print(max_heap_candidate_top)
        # print(max_heap_candidate_top)
        probability = -max_heap_candidate_top[0]
        # logit = -max_heap_candidate_top[0]
        indices = max_heap_candidate_top[2][1]
        mask = max_heap_candidate_top[2][2]
        index_to_expand = max_heap_candidate_top[2][0]

        suffix = "#"
        # print(index_to_word(indices[int(index_to_expand.numpy())], data))
        if (index_to_word(indices[int(index_to_expand.numpy())], data)).endswith(suffix):
            list = indices.shape.as_list()
            if int(index_to_expand.numpy()) == list[0] - 1:
                # print("Reach end")
                # min_heap_result.put((logit, indices, mask))
                min_heap_result.put((probability, indices, mask))
                if min_heap_result.qsize() > K:
                    min_heap_result.get()
            else:
                min_heap_result_top = min_heap_result.get()
                min_heap_result.put(min_heap_result_top)
                if min_heap_result_top[0] < probability:
                    index_tf = indices[:-1]
                    mask_tf = mask
                    expand_index_tf = tf.expand_dims(index_tf, 0)
                    expand_mask_tf = tf.expand_dims(mask_tf, 0)
                    preds = model(expand_index_tf)
                    (k_indices_tf, logits, probabilities, index_to_expand) = generate_K_indices_pq(preds,
                                                                                                   expand_mask_tf,
                                                                                                   data, K)
                    for i in range(K):
                        new_index_tf = tf.Variable(indices)
                        new_index_tf = new_index_tf[int(index_to_expand.numpy())].assign(
                            k_indices_tf[i, int(index_to_expand.numpy() - 1)])
                        new_mask_tf = tf.Variable(mask_tf)
                        new_mask_tf = new_mask_tf[int(index_to_expand.numpy())].assign(
                            tf.constant(1.0, dtype="float32"))
                        # print(input_indices_tf)
                        # print(mask_tf)
                        # print(new_index_tf)
                        # print(new_mask_tf)
                        count = count + 1
                        max_heap_candidate.put(
                            (-(probability * probabilities[i]), count, (index_to_expand, new_index_tf, new_mask_tf)))
                else:
                    break
        else:
            min_heap_result.put((probability, indices, mask))
            # print("We got one!!!!!")
            if min_heap_result.qsize() > K:
                min_heap_result.get()
    return min_heap_result


def beam_search(model, data, index_tf, mask_tf, K, input_logits, depth):
    result_indices = []
    result_logits = []

    if depth <= K:
        expand_index_tf = tf.expand_dims(index_tf, 0)
        expand_mask_tf = tf.expand_dims(mask_tf, 0)
        preds = model(expand_index_tf)
        # , k_indices_np, k_logits_np
        (k_indices, k_logits) = generate_K_indices(preds, expand_mask_tf, data, K, input_logits)
        index_to_expand = tf.reduce_sum(mask_tf) - 1
        for i in range(K):
            indices = k_indices[i]
            logits = k_logits[i]
            last_word_index = indices[int(index_to_expand.numpy())]
            last_word = data.i2w[last_word_index.numpy()]
            print(last_word)
            suffix = "#"
            if last_word.endswith(suffix) and depth < K:
                print("#####")
                new_index_tf = tf.Variable(index_tf)
                print("The new index:")
                print(new_index_tf)
                print(int(index_to_expand.numpy()) + 1)

                new_index_tf = new_index_tf[int(index_to_expand.numpy()) + 1].assign(last_word_index)
                new_mask_tf = tf.Variable(mask_tf)
                new_mask_tf = new_mask_tf[int(index_to_expand.numpy()) + 1].assign(tf.constant(1.0, dtype="float32"))

                (r_indices, r_logits) = beam_search(model, data, new_index_tf, new_mask_tf, K, logits, depth + 1)
                result_indices.extend(r_indices)
                result_logits.extend(r_logits)
            else:
                print("!!!!!")
                print(K)
                new_index_tf = tf.Variable(index_tf)
                new_index_tf = new_index_tf[int(index_to_expand.numpy()) + 1].assign(last_word_index)
                result_indices.append(new_index_tf)
                result_logits.append(logits)
                # if i is Ending or i size is ending:
                #     result_indices.add(i,logits)
                # else:
                #     beam_search(model, i, shifted_mask, K, result_indices)
    if depth == 0:
        for i in range(len(result_logits)):
            print("Print beam search result:")
            print(result_indices[i])
            indices_to_word2(result_indices[i].numpy(), data)
            print("Print beam search logits result:")
            print(result_logits[i])
        print(len(result_logits))
    return result_indices, result_logits


def generate_indices_masks(data, size):
    masks_result = []
    indices_result = []
    # print("SHAPE")
    # print(np.shape(data.valid_data))
    # print("SHAPE")
    random.Random(1032).shuffle(data.valid_data)
    # save not shuffle, need a manual seed
    for indices, masks in data.batcher(data.valid_data, is_training=False):
        indices_result.extend(indices[:size, :])
        masks_result.extend(masks[:size, :])
        break
    return indices_result, masks_result


def process_data(data, indices, mask):
    indices_result = []
    masks_result = []
    cur_indices = []
    cur_mask = []
    j = 0
    for i in range(201):
        if mask[i].numpy() == 0:
            break
        cur_indices.append(int(indices[i].numpy()))
        cur_mask.append(float(mask[i].numpy()))

        if not data.i2w[int(indices[i].numpy())].endswith("#"):
            indices_result.append(cur_indices.copy())
            # print(cur_indices.copy())
            masks_result.append(cur_mask.copy())

    # print(indices_result)
    return indices_result, masks_result


def get_score(pq, input_indices_tf_next, input_mask_tf_next, K, data):
    base = K + 1
    while not pq.empty():
        base = base - 1
        [probability, indices, masks] = pq.get()
        # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # print(indices)
        # indices_to_word2(indices.numpy(),data)
        # print(input_indices_tf_next)
        # indices_to_word(input_indices_tf_next,data)
        listA = input_indices_tf_next.shape.as_list()
        same = True
        # print(indices)
        # print(input_indices_tf_next)
        for i in range(listA[0]):
            # print(int(indices[i].numpy()))
            # print(int(input_indices_tf_next[i].numpy()))
            if not int(indices[i].numpy()) == int(input_indices_tf_next[i].numpy()):
                same = False
                break
        if same:
            # print("find one same")
            return 1/base
        # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    return 0


def my_test(model, data):
    indices = [0, 1684, 380, 704, 1210, 626, 1373, 34, 175, 1885, 1941, 1169, 642, 434]
    indices_to_word2(indices, data)
    masks = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0]
    # indices = [0, 2, 5, 3]
    # masks = [1.0, 1.0, 0, 0]
    input_indices_tf = tf.constant(indices, dtype="int32")
    input_mask_tf = tf.constant(masks, dtype="float32")
    K = 3
    # q = beam_search_pq(model, data, input_indices_tf, input_mask_tf, K)
    # while not q.empty():
    #     print(q.get())
    # for indices, masks in data.batcher(data.train_data, is_training=False):
    #     index = indices[5,:]
    #     mask = masks[5,:]
    #     print(index)
    #     print(mask)
    #     break
    test_data_size = 1
    test_count = 0
    test_socre = 0

    [indicess, maskss] = generate_indices_masks(data, test_data_size)
    for i in range(test_data_size):
        print(i)
        masks = maskss[i]
        indices = indicess[i]
        # print(indices)
        # print(masks)
        # indices_to_word(indices, data)
        [index, mask] = process_data(data, indices, masks)
        ll = len(index)
        # ll - 5
        for j in range(3, ll - 5):
            ii = index[j].copy()
            mm = mask[j].copy()
            test_count = test_count + 1;
            zero_to_add = len(index[j + 1]) - len(index[j]) + 5
            # add 5 because we need some extra room but not too much
            ii.extend([0] * zero_to_add)
            mm.extend([0.0] * zero_to_add)
            input_indices_tf = tf.constant(ii, dtype="int32")
            input_mask_tf = tf.constant(mm, dtype="float32")
            # print(ii)
            pq = beam_search_pq(model, data, input_indices_tf, input_mask_tf, K)
            test_socre = test_socre + get_score(pq, tf.constant(index[j + 1], dtype="int32"), tf.constant(mask[j + 1], dtype="float32"), K, data)
        MPR = test_socre / test_count
        print(MPR)
    MPR = test_socre/test_count
    print(MPR)


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

    # random.shuffle(data.valid_data)  # Shuffle just once
    # train(model, data)
    # # model.save_weights('test_model')
    # model.save_weights('test_model_2')
    # # f = open('store.pckl', 'wb')
    # f = open('store_2.pckl', 'wb')
    # pickle.dump(data, f)
    # f.close()

    f = open('store_1.pckl', 'rb')
    data = pickle.load(f)
    f.close()
    model.load_weights(os.path.join(os.sep, "home", "program", "code", "test_model_1"))
    my_test(model, data)

    # entropy, count = eval(model, data)
    # print("Validation: tokens: {0}, entropy: {1:.3f}, perplexity: {2:.3f}".format(count, entropy,
    #                                                                               0.0 if entropy > 100 else math.pow(2,
    #                                                                                                                  entropy)))

    print("Everything Done")


if __name__ == '__main__':
    main()
