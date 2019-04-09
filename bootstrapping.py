import torch
import torch.nn as nn
import json
import random


class Bootstrapping:
    def __init__(self, training_size, n_h):
        with open("word_embedding.json", 'r') as f:
            self.word_dict = json.load(f)
        self.diversity_phrases = set()
        self.test_dic = self.load_test()
        self.training_size = training_size
        self.training_dic = self.load_training()
        self.iteration = 0
        self.n_h = n_h
        self.train_list = list(self.training_dic.items())
        self.test_list = list(self.test_dic.items())
        self.train_input = [list(x[0]) for x in self.train_list]
        self.train_output = [[x[1]] for x in self.train_list]
        self.test_input = [list(x[0]) for x in self.test_list]
        self.test_output = [[x[1]] for x in self.test_list]
        self.x_test = torch.tensor(self.test_input)
        self.y_test = torch.tensor(self.test_output)
        self.iteration = 0
        self.n_in = 600
        self.n_out = 1
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.n_in, self.n_h),
            torch.nn.ReLU(),
            torch.nn.Linear(self.n_h, self.n_out),
            torch.nn.Sigmoid()
        )

    def load_test(self):
        f = open("test.txt", "r")
        test_dic = {}
        count = 0
        for p in f:
            count += 1
            if p[-1] == '\n':
                p = p[:-1]
            ws = p.split()
            emb = self.word_dict[ws[0]].copy()
            emb.extend(self.word_dict[ws[1]])
            if count <= 80:
                test_dic[tuple(emb)] = 1
                self.diversity_phrases.add(p)
            else:
                test_dic[tuple(emb)] = 0
        return test_dic

    def load_training(self):
        train_dic = {}
        f1 = open("related.txt", 'r')
        for p in f1:
            if p[-1] == '\n':
                p = p[:-1]
            ws = p.split()
            if ws[0] not in self.word_dict or ws[1] not in self.word_dict:
                continue

            emb = self.word_dict[ws[0]].copy()
            #    print(len(emb))
            emb.extend(self.word_dict[ws[1]])
            #    print(len(emb))
            train_dic[tuple(emb)] = 1.0
            self.diversity_phrases.add(p)

        f2 = open("unrelated.txt", 'r')
        for p in f2:
            if p[-1] == '\n':
                p = p[:-1]
            ws = p.split()
            if ws[0] not in self.word_dict or ws[1] not in self.word_dict:
                continue
            emb = self.word_dict[ws[0]].copy()
            emb.extend(self.word_dict[ws[1]])
            train_dic[tuple(emb)] = 0.0
            if len(train_dic) >= self.training_size:
                break
        return train_dic

    def shuffle(self):
        self.train_list = list(self.training_dic.items())
        random.shuffle(self.train_list)
        self.train_input = [list(x[0]) for x in self.train_list]
        self.train_output = [[x[1]] for x in self.train_list]

    def train(self):
        x = torch.tensor(self.train_input)
        y = torch.tensor(self.train_output)



        # Use the nn package to define our model and loss function.

        loss_fn = torch.nn.MSELoss(reduction='sum')
        learning_rate = 1e-4
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        for t in range(500):
            # Forward pass: compute predicted y by passing x to the model.
            y_pred = self.model(x)

            # Compute and print loss.
            loss = loss_fn(y_pred, y)
            #    print(t, loss.item())

            # Before the backward pass, use the optimizer object to zero all of the
            # gradients for the variables it will update (which are the learnable
            # weights of the model). This is because by default, gradients are
            # accumulated in buffers( i.e, not overwritten) whenever .backward()
            # is called. Checkout docs of torch.autograd.backward for more details.
            optimizer.zero_grad()

            # Backward pass: compute gradient of the loss with respect to model
            # parameters
            loss.backward()

            # Calling the step function on an Optimizer makes an update to its
            # parameters
            optimizer.step()

    def acc(self, thre=0.5):
        y_p = self.model(self.x_test)
        #    print (y_p)
        pos = 0
        t_pos = 0.0
        f_pos = 0.0
        cnt = 0.0
        for i in range(len(y_p)):
            if y_p[i][0] >= thre:
                res = 1

                if self.y_test[i][0] == 1:
                    t_pos += 1
                else:
                    f_pos += 1
            else:
                res = 0

            if res == self.y_test[i][0]:
                cnt += 1
            if self.y_test[i][0] == 1:
                pos += 1
        precision = float(t_pos / (t_pos + f_pos))
        recall = float(t_pos / pos)
        F1 = 2 * recall * precision / (precision + recall)
        acc = float(cnt / len(y_p))
        # print("Accuracy: ", acc)
        # print("Precision: ", precision)
        # print("Recall: ", recall)
        # print("F1: ", F1)
        return acc, precision, recall, F1

    def find_new(self, threshold=0.9):
        f3 = open("cand.txt", 'r')
        new_cnt = 0

        for p in f3:
            if p[-1] == '\n':
                p = p[:-1]
            ws = p.split()
            if p in self.diversity_phrases or ws[0] not in self.word_dict or ws[1] not in self.word_dict:
                continue
            emb = self.word_dict[ws[0]].copy()
            #    print(len(emb))
            emb.extend(self.word_dict[ws[1]])
            #    print(len(emb))
            x = torch.tensor([emb])
            y = self.model(x)
            if y[0][0] > threshold:
                new_cnt += 1
                self.diversity_phrases.add(p)
                self.training_dic[tuple(emb)] = 1.0
                print(p)
        #       print(p, float(y[0][0]))

        print("Add ", new_cnt, ' new diversity phrases.')
        return new_cnt

    def run_iteration(self, expected_precision = 0.95):
        print()
        print("Iteration ", self.iteration)
        self.iteration += 1
        self.shuffle()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.n_in, self.n_h),
            torch.nn.ReLU(),
            torch.nn.Linear(self.n_h, self.n_out),
            torch.nn.Sigmoid()
        )
        self.train()
        res = self.acc()
        print("Accuracy: ", res[0])
        print("Precision: ", res[1])
        print("Recall: ", res[2])
        print("F1: ", res[3])
        thre = 0.5
        prec = res[1]
        while prec < expected_precision and thre < 1.0:
            thre += 0.01
            prec = self.acc(thre)[1]
        print("When threshold is ", thre, ", precision will be ",prec)
        print ("Add phrases whose score is larger than ", thre)
        count = self.find_new(prec)
        return count

    def auto_bt(self, expected_precision = 0.95):
        new_cnt = self.run_iteration(expected_precision)
        while new_cnt > 0:
            new_cnt = self.run_iteration(expected_precision)

        print ("Total: ", len(self.diversity_phrases))
        f4 = open("dic_v3_1.txt", 'w')
        for p in self.diversity_phrases:
            f4.write(p)
            f4.write('\n')








b = Bootstrapping(5000, 200)
b.auto_bt(0.93)