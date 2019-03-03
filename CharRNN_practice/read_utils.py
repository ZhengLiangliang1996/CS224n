#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 12:01:58 2019

@author: liangliang
"""

import numpy as np
import copy 
import time 
import tensorflow as tf 
import pickle


def batch_generator(arr, n_seqs, n_steps):
    '''
    Para: 
         arr:     training text array(data)
         n_seqs:  number of sequence 
         n_steps: number of steps
         
    part of the explanation :
        https://cloud.tencent.com/developer/article/1019931
        每n_steps进一次循环（即生成一个batch），x就是提取出相应那一段的内容，np.zeros_like
        用于生成大小一致的tensor但所有元素全为0，然后将x的第一个元素放到y最后，其他位元素位置
        往前顺移一位赋给y。这x就代表了输入，而y就是有监督训练的标签（每个字符做预测时的正确答
        案就是文本的下一个字符）。yield的使用是将函数作为生成器，这样做省内存。
    '''
    
    arr = copy.copy(arr)                    # shallow copy : do affect the original arr
    batch_size = n_seqs * n_steps
    n_batches = int(len(arr) / batch_size)  # leave the rest data: using int
    arr = arr[:batch_size * n_batches]
    arr = arr.reshape((n_seqs, -1))         # flatten
    
    
    while True:
        np.random.shuffle(arr)
        for n in range(0, arr.shape[1], n_steps):
            x = arr[:, n:n + n_steps]
            y = np.zeros_like(x)
            y[:, :-1], y[:, -1] = x[:, 1:], x[:, 0]
            yield x, y

class TextConverter(object):
    def __init__(self, text=None, max_vocab=5000, filename=None):
        if filename is not None:
            with open(filename, 'rb') as f:
                self.vocab = pickle.load(f)
        else:
            vocab = set(text)             # delete the repetition
            print(len(vocab))
            
            vocab_count = {}
            for word in vocab:
                vocab_count[word] = 0
            for word in text:
                vocab_count[word] += 1
            vocab_count_list = []
            for word in vocab_count:
                vocab_count_list.append((word, vocab_count[word]))
            vocab_count_list.sort(key=lambda x: x[1], reverse=True)
            if len(vocab_count_list) > max_vocab:
                vocab_count_list = vocab_count_list[:max_vocab]
            vocab = [x[0] for x in vocab_count_list]
            self.vocab = vocab
        
        self.word_to_int_table = {c: i for i, c in enumerate(self.vocab)}
        self.int_to_word_table = dict(enumerate(self.vocab))
        
    @property
    def vocab_size(self):
        return len(self.vocab) + 1
    
    def word_to_int(self, word):
        if word in self.word_to_int_table:
            return self.word_to_int_table[word]
        else:
            return len(self.vocab)
    
    def int_to_word(self, index):
        if index == len(self.vocab):
            return '<unk>'
        elif index < len(self.vocab):
            return self.int_to_word_table[index]
        else:
            raise Exception('Unknown index!')
    
    def text_to_arr(self, text):
        arr = []
        for word in text:
            arr.append(self.word_to_int(word))
        return np.array(arr)
    
    def arr_to_text(self, arr):
        words = []
        for index in arr:
            words.append(self.int_to_word(index))
        return "".join(words)

    def save_to_file(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.vocab, f)
    
    
        
            





        
    
    
    
