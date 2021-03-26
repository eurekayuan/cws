import codecs
import os
import pickle
import json

def convert_charLen2one(char):
    if char == '<NUM>':
        char = 'N'
    elif char == '<ENG>':
        char = 'E'
    if len(char) != 1:
        char = 'U'
    else:
        char = char
    return char

def get_word(seq_chars, seq_tags):
    e_idx = 'E'
    s_idx = 'S'
    tag_seqs_sent = []
    word_seqs_sent = []
    tag_seqs = []
    word_seqs = []
    charsss = []
    for seq_char, seq_tag in zip(seq_chars, seq_tags):
        tag_seq = []
        word_seq = []
        start = 0
        for i, (char, tok) in enumerate(zip(seq_char, seq_tag)):
            char = convert_charLen2one(char)
            charsss.append(char)
            if tok == e_idx or tok == s_idx:
                word = seq_char[start: i + 1]
                tag = seq_tag[start: i + 1]
                start = i + 1
                word_seq.append(''.join(word))
                tag_seq.append(''.join(tag))
        if seq_char[-1] not in [e_idx,s_idx]:
            if start != len(seq_char):
                word = seq_char[start:]
                tag = seq_tag[start:]
                word_seq.append(''.join(word))
                tag_seq.append(''.join(tag))
        tag_seqs_sent.append(tag_seq)
        word_seqs_sent.append(word_seq)
        tag_seqs += tag_seq
        word_seqs += word_seq
    count =0
    for word,tag in zip(word_seqs,tag_seqs):
        for char,t in zip(word,tag):
            count+=1
            if count == len(charsss):
                break
        if count== len(charsss):
            break

    return word_seqs,tag_seqs,word_seqs_sent,tag_seqs_sent

def read_data(corpus_type, fn):
    mode = '#'
    column_no = -1

    src_data = []
    data = []
    label = []
    data_sent =[]
    label_sent =[]

    src_data_sentence = []
    data_sentence = []
    label_sentence = []

    with codecs.open(fn, 'r', 'utf-8') as f:
        lines = f.readlines()
    for line in lines:
        line_t = line.replace('\n', '').replace('\r', '').replace('  ', '#').split('#')
        if len(line_t) < 3:
            if len(data_sentence) == 0:
                continue
            src_data.append(src_data_sentence)
            data.append(data_sentence)
            label.append(label_sentence)
            src_data_sentence = []
            data_sentence = []
            label_sentence = []
            continue
        src_word =line_t[0]
        word = convert_charLen2one(line_t[1])
        src_data_sentence.append(src_word)
        data_sentence.append(word)
        data_sent.append(word)
        label_sent.append(line_t[2].split('_')[0])
        label_sentence += [line_t[2].split('_')[0]]

    return data_sent, label_sent, data, label

def read_data_test2(corpus_type, fn_test_result, fn_test):
    words_sent = []
    truetags_sent =[]
    predtags_sent=[]

    words_all = []
    truetags_all = []
    predtags_all = []

    words = []
    truetags = []
    predtags = []

    word1=[]
    truetag1 =[]
    predtag1 =[]
    count =-1
    with codecs.open(fn_test_result, 'r', 'utf-8') as f:
        lines = f.readlines()
    for line in lines:

        line_t = line.replace('\n', '').replace('\r', '').replace('  ', '#').strip().split()
        if len(line_t) ==0:
            count += 1
            # if count > 5:
            # 	break
            words_sent.append(word1)
            truetags_sent.append(truetag1)
            predtags_sent.append(predtag1)
            words += word1
            truetags += truetag1
            predtags += predtag1

            word1 = []
            truetag1 = []
            predtag1 = []
        else:
            if len(line_t)==2:
                w1='U'
                t1 =line_t[0]
                t2 =line_t[1]
            else:
                w1 = convert_charLen2one(line_t[0])
                t1 =line_t[1]
                t2 =line_t[2]
            word1.append(w1)
            truetag1.append(t1)
            predtag1.append(t2)

    for ws,ts,ps in zip(words_sent,truetags_sent,predtags_sent):
        words_all += ws
        truetags_all += ts
        predtags_all += ps



    words_final =[]
    words_sent_final =[]
    count_testwordNotEqual2Original =0
    count=0
    data_test, label_test, data_sent_test, label_sent_test = read_data(corpus_type, fn_test)
    for wt_s,ws,ts,ps in zip(data_sent_test,words_sent,truetags_sent,predtags_sent):
        new_ws =[]
        # print(len(wt_s),len(ws),len(ts),len(ps))
        for wt,w,t,p in zip(wt_s,ws,ts,ps):
            count+=1
            if wt !=w:
                w= wt
                count_testwordNotEqual2Original+=1
            words_final.append(w)
            new_ws.append(w)
        words_sent_final.append(new_ws)
    
    return words_all, truetags_all, predtags_all, words_sent, truetags_sent, predtags_sent

def get_words_num(word_sequences):
    return sum(len(word_seq) for word_seq in word_sequences)

def span_entity_amb_fre(word_seqs_train_sent,tag_seqs_train_sent, fnwrite_GoldenWordHard):
    if os.path.exists(fnwrite_GoldenWordHard):
        fread = open(fnwrite_GoldenWordHard, 'rb')
        Hard_cwsWord_inTrain, Freq_cwsWord_inTrain_normalize = pickle.load(fread)
        return Hard_cwsWord_inTrain, Freq_cwsWord_inTrain_normalize

def span_token_amb_fre(train_seqchar, train_seqtag, fnwrite_GoldenCharHard):
    if os.path.exists(fnwrite_GoldenCharHard):
        fread = open(fnwrite_GoldenCharHard, 'rb')
        Hard_cwsChar_inTrain, Freq_cwsChar_inTrain_normalize = pickle.load(fread)
        return Hard_cwsChar_inTrain, Freq_cwsChar_inTrain_normalize

def span_sent_oovDen_length(train_vocab,test_word_sequences_sent,fnwrite_sentLevelDensity):
    if os.path.exists(fnwrite_sentLevelDensity):
        fread = open(fnwrite_sentLevelDensity, 'rb')
        max_wordLength, oov_density, sent_len = pickle.load(fread)
        return max_wordLength, oov_density, sent_len

def precompute(corpus_type, train_vocab, fn_train, fn_test_results, fn_test, fnwrite_GoldenWordHard, fnwrite_GoldenCharHard, fnwrite_sentLevelDensity):
    train_seqchar, train_seqtag, train_seqchar_sent, train_seqtag_sent = read_data(corpus_type, fn_train)
    test_seqchar, test_truetags, test_predtags, test_seqchar_sent, test_truetags_sent, test_predtags_sent = read_data_test2(corpus_type, fn_test_results, fn_test)

    word_seqs_train, tag_seqs_train, word_seqs_train_sent, tag_seqs_train_sent = get_word(train_seqchar_sent, train_seqtag_sent)
    word_true_test, truetag_test, word_true_test_sent, truetag_test_sent = get_word(test_seqchar_sent, test_truetags_sent)
    word_pred_test, predtag_test, word_pred_test_sent, predtag_test_sent = get_word(test_seqchar_sent, test_predtags_sent)
    preCompute_ambSpan, preCompute_freqSpan = span_entity_amb_fre(word_seqs_train_sent, tag_seqs_train_sent, fnwrite_GoldenWordHard)
    preCompute_ambToken, preCompute_freqToken = span_token_amb_fre(train_seqchar, train_seqtag, fnwrite_GoldenCharHard)
    max_wordLength, preCompute_odensity, preCompute_slength = span_sent_oovDen_length(train_vocab, test_seqchar_sent, fnwrite_sentLevelDensity)
    return preCompute_ambSpan, preCompute_freqSpan, preCompute_ambToken, preCompute_freqToken, preCompute_odensity, preCompute_slength

def load():
    with open('config.json', 'r') as config:
        params = json.load(config)
    path_data = params['path_data']
    task_type = params['task_type']
    corpus_type = params['corpus_type']
    path_preComputed = params['path_preComputed']
    fn_test_results = params['fn_test_results']
    fn_train = path_data + task_type + "/data_" + corpus_type + '/train'
    fn_test  = path_data + task_type + "/data_" + corpus_type + '/test'
    train_vocab 			 = path_preComputed + corpus_type + '_vocab.txt'
    fnwrite_GoldenWordHard 	 = path_preComputed + corpus_type + '_rhoWordHardFreq.pkl'
    fnwrite_GoldenCharHard 	 = path_preComputed + corpus_type + '_rhoCharHardFreq.pkl'
    fnwrite_sentLevelDensity = path_preComputed + corpus_type + '_sentLevelDensity.pkl'
    wCon, wFre, cCon, cFre, oDen, sLen = precompute(corpus_type, train_vocab, fn_train, fn_test_results, fn_test, fnwrite_GoldenWordHard, fnwrite_GoldenCharHard, fnwrite_sentLevelDensity)
    return wCon, wFre, cCon, cFre, oDen, sLen

if __name__ == '__main__':
    load()