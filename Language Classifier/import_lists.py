import sys
import numpy as np

norm_val = .1

def import_lists(file1, file2):
    fname1 = "../Word Lists/" + file1
    fname2 = "../Word Lists/" + file2
    lang1 = open(fname1, "r")
    lang2 = open(fname2, "r")
    wlist1 = []
    wlist2 = []

    for word in lang1:
        wlist1.append(word)

    for word in lang2:
        wlist2.append(word)

    wrapped_list = [wlist1, wlist2]

    return wrapped_list

def select_random(wrapped_list):
    choose_list = np.random.randint(2, size=1)[0]
    list_len = len(wrapped_list[choose_list])
    choose_word = np.random.randint(list_len, size=1)[0]
    chosen = [wrapped_list[choose_list][choose_word], choose_list]
    return chosen

def to_vector(word):
    vector = []
    word = word.strip("\n")

    for c in word:
        normal_c = ord(c) - 97
        for hit in range(26):
            if hit == normal_c:
                vector.append(norm_val)
            else:
                vector.append(0)

    for diff in range(15 - len(word)):
        for i in range(26):
            vector.append(0)

    return vector

def to_word(vector):
    str = ""
    offset = 0

    for c in vector:
        if c == norm_val:
            outchar = chr(ord('a') + offset)
            str += outchar
        offset += 1
        if offset == 26:
            offset = 0

    return str

def training_pair(wrapped_list):
    choice = select_random(wrapped_list)
    word = choice[0]
    targ = 0
    if choice[1] == 0:
        targ = -1
    else:
        targ = 1
    return [to_vector(word), targ]
