import numpy as np
import codecs
import os
import matplotlib.pyplot as plt
import matplotlib.figure
import pickle
import loadAttribute

debug = False

class Evaluator:
    def __init__(self):
        self.result_dir = 'ctb_results_8models'
        self.wCon, self.wFre, self.cCon, self.cFre, self.oDen, self.sLen = loadAttribute.load()

    # convert special tokens
    def __lbl2char(self, char):
        if char == '<NUM>':
            char = 'N'
        elif char == '<ENG>':
            char = 'E'
        if len(char) != 1:
            char = 'U'
        else:
            char = char
        return char

    # chunk: tuple of word, start and end
    def __getChunks(self, word_seqs):
        chunks = []
        chunk_start = 0
        for word in word_seqs:
            chunk = (word, chunk_start, chunk_start + len(word))
            chunks.append(chunk)
            chunk_start = chunk_start + len(word)
        return chunks

    def __checkChunksRecall(self, pred_chunks, true_chunks):
        c = []
        successful_predict = set(true_chunks) & set(pred_chunks)
        for i in range(0, len(true_chunks)):
            current_chunk = true_chunks[i]
            if current_chunk in successful_predict:
                correctness = 1
            else:
                correctness = 0
            c.append(correctness)
        return c

    def __checkChunksPrecision(self, pred_chunks, true_chunks):
        c = []
        successful_predict = set(true_chunks) & set(pred_chunks)
        for i in range(0, len(pred_chunks)):
            current_chunk = pred_chunks[i]
            if current_chunk in successful_predict:
                correctness = 1
            else:
                correctness = 0
            c.append(correctness)
        return c
    
    def __readwseqs(self, file_name):
        lines_t = []
        wseqs_true = []
        wseqs_pred = []
        tseqs_true = []
        tseqs_pred = []
        with open(file_name, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if len(line) < 3:
                    continue
                elements = line.split(' ')
                elements[0] = self.__lbl2char(elements[0])
                lines_t.append(elements)
        current_word = ''
        current_tag = ''
        for i in range(0, len(lines_t)):
            if lines_t[i][1] == 'S':
                if current_word != '':
                    wseqs_true.append(current_word)
                    tseqs_true.append(current_tag)
                wseqs_true.append(lines_t[i][0])
                tseqs_true.append(lines_t[i][1])
                current_word = ''
                current_tag = ''
            elif lines_t[i][1] == 'B':
                if current_word != '':
                    wseqs_true.append(current_word)
                    tseqs_true.append(current_tag)
                current_word = lines_t[i][0]
                current_tag = lines_t[i][1]
            elif lines_t[i][1] == 'M':
                current_word += lines_t[i][0]
                current_tag += lines_t[i][1]
            elif lines_t[i][1] == 'E':
                current_word += lines_t[i][0]
                current_tag += lines_t[i][1]
                wseqs_true.append(current_word)
                tseqs_true.append(current_tag)
                current_word = ''
                current_tag = ''
        current_word = ''
        current_tag = ''
        for i in range(0, len(lines_t)):
            if lines_t[i][2] == 'S':
                if current_word != '':
                    wseqs_pred.append(current_word)
                    tseqs_pred.append(current_tag)
                wseqs_pred.append(lines_t[i][0])
                tseqs_pred.append(lines_t[i][2])
                current_word = ''
                curent_tag = ''
            elif lines_t[i][2] == 'B':
                if current_word != '':
                    wseqs_pred.append(current_word)
                    tseqs_pred.append(current_tag)
                current_word = lines_t[i][0]
                current_tag =lines_t[i][2]
            elif lines_t[i][2] == 'M':
                current_word += lines_t[i][0]
                current_tag += lines_t[i][2]
            elif lines_t[i][2] == 'E':
                current_word += lines_t[i][0]
                current_tag += lines_t[i][2]
                wseqs_pred.append(current_word)
                tseqs_pred.append(current_tag)
                current_word = ''
                current_tag = ''
        return wseqs_pred, wseqs_true, tseqs_pred, tseqs_true

    # generate figure
    def show(self, results, scores, title='Balanced recall, precision, f1-score with difficulty'):
        rs = []
        ps = []
        fs = []
        for score in scores:
            rs.append(score[0])
            ps.append(score[1])
            fs.append(score[2])
        xs = np.arange(len(results))
        fig = plt.figure()
        f = fig.add_subplot(111)
        f.bar(xs - 0.2, rs, label='balanced recall', width=0.2, color='aqua')
        f.bar(xs, ps, label='balanced precision', width=0.2, color='orange')
        f.bar(xs + 0.2, fs, label='balanced f1-score', width=0.2, color='cornflowerblue')
        f.set_xticks(xs)
        f.set_xticklabels(results)
        f.set_xlabel('model results')
        f.set_ylabel('balanced scores')
        f.set_title(title)
        f.legend()
        f.figure.autofmt_xdate()
        plt.show()

    def evaluate_model_traditional(self, pred_chunks, true_chunks):
        if debug == True:
            print('len(pred_chunks)', len(pred_chunks))
            print('len(true_chunks)', len(true_chunks))
        correct_preds, total_correct, total_preds = 0., 0., 0.
        correct_preds += len(set(true_chunks) & set(pred_chunks))
        total_preds += len(pred_chunks)
        total_correct += len(true_chunks)
        if debug == True:
            print('correct_preds', correct_preds)
        p = correct_preds / total_preds if correct_preds > 0 else 0
        r = correct_preds / total_correct if correct_preds > 0 else 0
        f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0
        return r, p, f1

    def evaluate_traditional(self):
        results = os.listdir(self.result_dir)
        scores = []
        for result in results:
            seqs_pred, seqs_true, tags_pred, tags_true = self.__readwseqs(self.result_dir + '/' + result)
            score = self.evaluate_model_traditional(self.__getChunks(seqs_pred), self.__getChunks(seqs_true))
            scores.append(score)
        return results, scores

    def __model_difficulty_golden(self, pred_chunks, true_chunks):
        d = []
        successful_predict = set(true_chunks) & set(pred_chunks)
        for i in range(0, len(true_chunks)):
            current_chunk = true_chunks[i]
            if current_chunk in successful_predict:
                difficulty = 0
            else:
                difficulty = 1
            d.append(difficulty)
        return d

    def __model_difficulty_pred(self, pred_chunks, true_chunks, d):
        d2 = []
        j = 0
        for i in range(0, len(pred_chunks)):
            while true_chunks[j][2] < pred_chunks[i][2]:
                j += 1
            d2.append(d[j])
        return d2
        
    def __average_difficulty(self, d_list):
        sum = 0
        for d in d_list:
            vec = np.array(d)
            sum += vec
        difficulty = sum / len(d_list)
        return list(difficulty)

    def __difficulty_psychometric(self):
        d_list = []

        # use results themselves as the committee
        committee_dir = self.result_dir

        results = os.listdir(committee_dir)
        for result in results:
            seqs_pred, seqs_true, tags_pred, tags_true = self.__readwseqs(committee_dir + '/' + result)
            d = self.__model_difficulty_golden(self.__getChunks(seqs_pred), self.__getChunks(seqs_true))
            d_list.append(d)
        difficulty = self.__average_difficulty(d_list)
        return difficulty

    def evaluate_model_psychometric(self, pred_chunks, true_chunks, difficulty):
        if debug == True:
            print('len(pred_chunks)', len(pred_chunks))
            print('len(true_chunks)', len(true_chunks))
        correct_preds, total_correct, total_preds = 0., 0., 0.
        correct_preds += len(set(true_chunks) & set(pred_chunks))
        total_preds += len(pred_chunks)
        total_correct += len(true_chunks)
        if debug == True:
            print('correct_preds', correct_preds)

        vec_m = np.array(self.__checkChunksRecall(pred_chunks, true_chunks))
        vec_d = np.array(difficulty)

        # balanced recall
        r_reward = np.sum(vec_d * vec_m) / np.sum(vec_d)
        r_punishment = np.sum((1 - vec_d) * vec_m) / np.sum((1 - vec_d))
        r_balanced = (2 * r_reward * r_punishment) / (r_reward + r_punishment)
        
        # balanced precision
        vec_d2 = np.array(self.__model_difficulty_pred(pred_chunks, true_chunks, difficulty))
        p_reward = np.sum(vec_d * vec_m) / np.sum(vec_d2)
        p_punishment = np.sum((1 - vec_d) * vec_m) / np.sum(1 - vec_d2)
        p_balanced = (2 * p_reward * p_punishment) / (p_reward + p_punishment)

        # balanced f1
        f1_balanced = (2 * p_balanced * r_balanced) / (p_balanced + r_balanced)
        
        return r_balanced, p_balanced, f1_balanced

    def evaluate_psychometric(self):
        difficulty = self.__difficulty_psychometric()
        results = os.listdir(self.result_dir)
        scores = []
        for result in results:
            seqs_pred, seqs_true, tags_pred, tags_true = self.__readwseqs(self.result_dir + '/' + result)
            score = self.evaluate_model_psychometric(self.__getChunks(seqs_pred), self.__getChunks(seqs_true), difficulty)
            scores.append(score)
        return results, scores

    def evaluate_model(self, pred_seqs, true_seqs, pred_tags, true_tags, fun4difficulty):
        pred_chunks = self.__getChunks(pred_seqs)
        true_chunks = self.__getChunks(true_seqs)
        if debug == True:
            print('len(pred_chunks)', len(pred_chunks))
            print('len(true_chunks)', len(true_chunks))
        correct_preds, total_correct, total_preds = 0., 0., 0.
        correct_preds += len(set(true_chunks) & set(pred_chunks))
        total_preds += len(pred_chunks)
        total_correct += len(true_chunks)
        if debug == True:
            print('correct_preds', correct_preds)
        d_recall, d_precision = fun4difficulty(pred_seqs, true_seqs, pred_tags, true_tags)
        
        # balanced recall
        vec_m = np.array(self.__checkChunksRecall(pred_chunks, true_chunks))
        vec_d = np.array(d_recall)
        r_reward = np.sum(vec_d * vec_m) / np.sum(vec_d)
        r_punishment = np.sum((1 - vec_d) * vec_m) / np.sum((1 - vec_d))
        r_balanced = (2 * r_reward * r_punishment) / (r_reward + r_punishment)
        
        # balanced precision
        vec_m2 = np.array(self.__checkChunksPrecision(pred_chunks, true_chunks))
        vec_d2 = np.array(d_precision)
        p_reward = np.sum(vec_d2 * vec_m2) / np.sum(vec_d2)
        p_punishment = np.sum((1 - vec_d2) * vec_m2) / np.sum(1 - vec_d2)
        p_balanced = (2 * p_reward * p_punishment) / (p_reward + p_punishment)

        # balanced f1
        f1_balanced = (2 * p_balanced * r_balanced) / (p_balanced + r_balanced)

        return r_balanced, p_balanced, f1_balanced



    # BEGIN EVALUATION WITH 7 ATTRIBUTES

    def __difficulty_wCon(self, pred_seqs, true_seqs, pred_tags, true_tags):
        d_recall = []
        d_precision = []
        for i in range(0, len(pred_seqs)):
            pred_word = pred_seqs[i]
            if pred_word in self.wCon.keys():
                wds = self.wCon[pred_word]
                if pred_tags[i] in wds:
                    d_precision.append((1. - wds[pred_tags[i]]))
                else:
                    d_precision.append(1.)
            else:
                d_precision.append(1.)
        for i in range(0, len(true_seqs)):
            true_word = true_seqs[i]
            if true_word in self.wCon.keys():
                wds = self.wCon[true_word]
                if true_tags[i] in wds:
                    d_recall.append((1. - wds[true_tags[i]]))
                else:
                    d_recall.append(1.)
            else:
                d_recall.append(1)
        return d_recall, d_precision

    def evaluate_wCon(self):
        results = os.listdir(self.result_dir)
        scores = []
        for result in results:
            seqs_pred, seqs_true, tags_pred, tags_true = self.__readwseqs(self.result_dir + '/' + result)
            score = self.evaluate_model(seqs_pred, seqs_true, tags_pred, tags_true, self.__difficulty_wCon)
            scores.append(score)
        return results, scores
    
    def __difficulty_wFre(self, pred_seqs, true_seqs, pred_tags, true_tags):
        d_recall = []
        d_precision = []
        for word in true_seqs:
            if word in self.wFre.keys():
                d_recall.append(1. - float(self.wFre[word]))
            else:
                d_recall.append(1.)
        for word in pred_seqs:
            if word in self.wFre.keys():
                d_precision.append(1. - float(self.wFre[word]))
            else:
                d_precision.append(1.)
        return d_recall, d_precision

    def evaluate_wFre(self):
        results = os.listdir(self.result_dir)
        scores = []
        for result in results:
            seqs_pred, seqs_true, tags_pred, tags_true = self.__readwseqs(self.result_dir + '/' + result)
            score = self.evaluate_model(seqs_pred, seqs_true, tags_pred, tags_true, self.__difficulty_wFre)
            scores.append(score)
        return results, scores

    def __difficulty_cCon(self, pred_seqs, true_seqs, pred_tags, true_tags):
        d_recall = []
        d_precision = []
        for i in range(0, len(true_seqs)):
            word = true_seqs[i]
            d_recall_elem = 0.
            for j in range(0, len(word)):
                char = word[j]
                if char in self.cCon.keys():
                    if true_tags[i][j] in self.cCon[char]:
                        d_recall_elem += 1. - self.cCon[char][true_tags[i][j]]
                    else:
                        d_recall_elem += 1.
                else:
                    d_recall_elem += 1.
            d_recall.append(d_recall_elem)
        for i in range(0, len(pred_seqs)):
            word = pred_seqs[i]
            d_pred_elem = 0.
            for j in range(0, len(word)):
                char = word[j]
                if char in self.cCon.keys():
                    if pred_tags[i][j] in self.cCon[char]:
                        d_pred_elem += 1. - self.cCon[char][pred_tags[i][j]]
                    else:
                        d_pred_elem += 1.
                else:
                    d_pred_elem += 1.
            d_precision.append(d_pred_elem)

        return d_recall, d_precision

    def evaluate_cCon(self):
        results = os.listdir(self.result_dir)
        scores = []
        for result in results:
            seqs_pred, seqs_true, tags_pred, tags_true = self.__readwseqs(self.result_dir + '/' + result)
            score = self.evaluate_model(seqs_pred, seqs_true, tags_pred, tags_true, self.__difficulty_cCon)
            scores.append(score)
        return results, scores

    def __difficulty_cFre(self, pred_seqs, true_seqs, pred_tags, true_tags):
        d_recall = []
        d_precision = []
        for word in true_seqs:
            d_recall_elem = 0.
            for char in word:
                if char in self.cFre.keys():
                    d_recall_elem += 1. - float(self.cFre[char])
                else:
                    d_recall_elem += 1.
            d_recall.append(d_recall_elem)
        for word in pred_seqs:
            d_pred_elem = 0.
            for char in word:
                if char in self.cFre.keys():
                    d_pred_elem += 1. - float(self.cFre[char])
                else:
                    d_pred_elem += 1.
            d_precision.append(d_pred_elem)
    
        return d_recall, d_precision
    
    def evaluate_cFre(self):
        results = os.listdir(self.result_dir)
        scores = []
        for result in results:
            seqs_pred, seqs_true, tags_pred, tags_true = self.__readwseqs(self.result_dir + '/' + result)
            score = self.evaluate_model(seqs_pred, seqs_true, tags_pred, tags_true, self.__difficulty_cFre)
            scores.append(score)
        return results, scores

    def __difficulty_wLen(self, pred_seqs, true_seqs, pred_tags, true_tags):
        d_recall = []
        d_precision = []
        maxLen = 0
        for word in true_seqs:
            d_recall.append(len(word))
            if len(word) > maxLen:
                maxLen = len(word)
        maxLen = 0
        for word in pred_seqs:
            d_precision.append(len(word))
            if len(word) > maxLen:
                maxLen = len(word)
        tmp1 = np.array(d_recall)
        tmp2 = np.array(d_precision)
        tmp1 = tmp1 / maxLen
        tmp2 = tmp2 / maxLen
        return tmp1.tolist(), tmp2.tolist()

    def evaluate_wLen(self):
        results = os.listdir(self.result_dir)
        scores = []
        for result in results:
            seqs_pred, seqs_true, tags_pred, tags_true = self.__readwseqs(self.result_dir + '/' + result)
            score = self.evaluate_model(seqs_pred, seqs_true, tags_pred, tags_true, self.__difficulty_wLen)
            scores.append(score)
        return results, scores

    def __difficulty_oDen(self, pred_seqs, true_seqs, pred_tags, true_tags):
        d_recall = []
        d_precision = []
        i = 0
        l = 0
        maxLen = 0
        for word in true_seqs:
            if l > self.sLen[i]:
                i += 1
                l = 0
            d_recall.append(self.oDen[i])
            l += len(word)
        i = 0
        l = 0
        maxLen = 0
        for word in pred_seqs:
            if l > self.sLen[i]:
                i += 1
                l = 0
            d_precision.append(self.oDen[i])
            l += len(word)
        return d_recall, d_precision

    def evaluate_oDen(self):
        results = os.listdir(self.result_dir)
        scores = []
        for result in results:
            seqs_pred, seqs_true, tags_pred, tags_true = self.__readwseqs(self.result_dir + '/' + result)
            score = self.evaluate_model(seqs_pred, seqs_true, tags_pred, tags_true, self.__difficulty_oDen)
            scores.append(score)
        return results, scores

    def __difficulty_sLen(self, pred_seqs, true_seqs, pred_tags, true_tags):
        d_recall = []
        d_precision = []
        i = 0
        l = 0
        maxLen = 0
        for word in true_seqs:
            if l > self.sLen[i]:
                i += 1
                l = 0
            d_recall.append(self.sLen[i])
            l += len(word)
            if len(word) > maxLen:
                maxLen = len(word)
        i = 0
        l = 0
        maxLen = 0
        for word in pred_seqs:
            if l > self.sLen[i]:
                i += 1
                l = 0
            d_precision.append(self.sLen[i])
            l += len(word)
            if len(word) > maxLen:
                maxLen = len(word)
        tmp1 = np.array(d_recall)
        tmp2 = np.array(d_precision)
        tmp1 = tmp1 / maxLen
        tmp2 = tmp2 / maxLen
        return tmp1.tolist(), tmp2.tolist()

    def evaluate_sLen(self):
        results = os.listdir(self.result_dir)
        scores = []
        for result in results:
            seqs_pred, seqs_true, tags_pred, tags_true = self.__readwseqs(self.result_dir + '/' + result)
            score = self.evaluate_model(seqs_pred, seqs_true, tags_pred, tags_true, self.__difficulty_sLen)
            scores.append(score)
        return results, scores
    

if __name__ == '__main__':
    evaluator = Evaluator()
    results, scores = evaluator.evaluate_oDen()
    evaluator.show(results, scores)

    

