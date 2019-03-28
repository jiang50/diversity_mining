import torch
import torch.nn as nn
import json
import random


with open("word_embedding.json", 'r') as f:
    word_dict = json.load(f)



# print(len(word_dict['diverse']))
# print(word_dict['workplace'])

related = []
unrelated = []
test_dic = {}
f = open("test.txt", "r")
count = 0
for p in f:
    count += 1
    if p[-1] == '\n':
        p = p[:-1]
    ws = p.split()
    emb = word_dict[ws[0]].copy()
    emb.extend(word_dict[ws[1]])
    if count <= 80:
        test_dic[tuple(emb)] = 1
    else:
        test_dic[tuple(emb)] = 0

train_dic = {}
f1 = open("related.txt", 'r')
for p in f1:
    if p[-1] == '\n':
        p = p[:-1]
    ws = p.split()
    if ws[0] not in word_dict or ws[1] not in word_dict:
        print(p)
        continue

#    print(p,ws[0],ws[1],len(word_dict[ws[0]]))
    emb = word_dict[ws[0]].copy()
#    print(len(emb))
    emb.extend(word_dict[ws[1]])
#    print(len(emb))
    train_dic[tuple(emb)] = 1.0



# for key in train_dic:
#     print (len(key))
f2 = open("unrelated.txt", 'r')
for p in f2:
    if p[-1] == '\n':
        p = p[:-1]
    ws = p.split()
    if ws[0] not in word_dict or ws[1] not in word_dict:
        continue
    emb = word_dict[ws[0]].copy()
    emb.extend(word_dict[ws[1]])
    train_dic[tuple(emb)] = 0.0
    if len(train_dic) >= 2500:
        break
#print(len(train_dic))

# for key in train_dic:
#     print (len(key))


train_list = list(train_dic.items())
#print(len(train_list[0][0]))
rand = [x for x in range(2500)]
random.shuffle(rand)
#print (rand)
train_input = []
train_output = []



for t in rand:
    train_input.append(list(train_list[t][0]))
    train_output.append([train_list[t][1]])
# print (len(train_input))
# print (len(train_input[0]))


test_list = list(test_dic.items())
#print(len(train_list[0][0]))
rand = [x for x in range(400)]
random.shuffle(rand)
#print (rand)
test_input = []
test_output = []

for t in rand:
    test_input.append(list(test_list[t][0]))
    test_output.append([float(test_list[t][1])])




n_in, n_h, n_out, batch_size = 600, 200, 1, 2500


x = torch.tensor(train_input)
y = torch.tensor(train_output)

x_test = torch.tensor(test_input)
y_test = torch.tensor(test_output)

#Use the nn package to define our model and loss function.
model = torch.nn.Sequential(
    torch.nn.Linear(n_in, n_h),
    torch.nn.ReLU(),
    torch.nn.Linear(n_h, n_out),
    torch.nn.Sigmoid()
)
loss_fn = torch.nn.MSELoss(reduction='sum')


def acc(x, y):
    y_p = model(x)
    cnt = 0.0
#    print (y_p)
    pos = 0
    t_pos = 0.0
    f_pos = 0.0
    cnt = 0.0
    for i in range(len(y_p)):
        if y_p[i][0] >= 0.5:
            res = 1

            if y[i][0] == 1:
                t_pos += 1
            else:
                f_pos += 1
        else:
            res = 0

        if res == y[i][0]:
            cnt += 1
        if y[i][0] == 1:
            pos += 1
    precision = float(t_pos / (t_pos + f_pos))
    recall = float(t_pos / pos)
    print("Accuracy: ", float(cnt / len(y_p)))
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1: ", 2 * recall * precision / (precision + recall))
    return float(cnt / len(y_p))


# Use the optim package to define an Optimizer that will update the weights of
# the model for us. Here we will use Adam; the optim package contains many other
# optimization algoriths. The first argument to the Adam constructor tells the
# optimizer which Tensors it should update.
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for t in range(500):
    # Forward pass: compute predicted y by passing x to the model.
    y_pred = model(x)

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

acc(x, y)

acc(x_test, y_test)

f3 = open("cand.txt", 'r')
new_cnt = 0
new_dic = set()
for p in f3:
    if p[-1] == '\n':
        p = p[:-1]
    ws = p.split()
    if ws[0] not in word_dict or ws[1] not in word_dict:
        continue
    emb = word_dict[ws[0]].copy()
    #    print(len(emb))
    emb.extend(word_dict[ws[1]])
    #    print(len(emb))
    x = torch.tensor([emb])
    y = model(x)
    if y[0][0] > 0.9:
        new_cnt += 1
        new_dic.add(p)
 #       print(p, float(y[0][0]))

print ("Total: ", new_cnt)
f4 = open("dic_v3_0.txt", 'w')
for p in new_dic:
    f4.write(p)
    f4.write('\n')