# Creating our own KNN algorithm
from math import sqrt
import warnings
import numpy as np
from collections import Counter
import pandas as pd
import random


# Creating a function for our knn algorithm
def k_nearest_neighbors(data, predict, k=3):
    if len(data) >= k:
        warnings.warn("K is set to a value more than total voting group")
    distances = []
    # Creating the for loop that iterates through every group in our data
    for group in data:
        # inner loop that iterates through every feature of our data
        for features in data[group]:
            # calculating euclidean distance of every points with data point to predict
            # and choosing the group which has mininum euclidean distance with the new point
            euclidean_distance = np.linalg.norm(np.array(features) - np.array(predict))
            distances.append([euclidean_distance, group])

    # Ranking the distance to choose the minimum one
    votes = [i[1] for i in sorted(distances)[:k]]
    print(Counter(votes).most_common(1))
    vote_result = Counter(votes).most_common(1)[0][0]
    return vote_result


df = pd.read_csv("breast-cancer-wisconsin2.txt")
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)
full_data = df.astype(float).values.tolist()
random.shuffle(full_data)

test_size = 0.2
train_set = {2: [], 4: []}
test_set = {2: [], 4: []}
train_data = full_data[:-int(test_size*len(full_data))]
test_data = full_data[-int(test_size*len(full_data)):]

for i in train_data:
    train_set[i[-1]].append(i[:-1])

for i in test_data:
    test_set[i[-1]].append(i[:-1])

correct = 0
total = 0

for group in test_set:
    for data in test_set[group]:
        vote = k_nearest_neighbors(train_set, data, k=5)
        if group == vote:
            correct += 1
        total += 1

print('Accuracy:', correct/total)








