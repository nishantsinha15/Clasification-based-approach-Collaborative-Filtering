import csv
import time
from sklearn import svm
import numpy as np

user_count = 943
items_count = 1682
rating_dataset = {}
item_feature = {}
user_feature = {}


def item_features():
    name1 = 'Dataset/u.item'
    item_vector = {}
    with open(name1, encoding="ISO-8859-1", newline='') as f:
        reader = csv.reader(f, delimiter='|')
        for row in reader:
            item_vector[int(row[0])] = np.asarray(list(map(int, row[-19:])))
    return item_vector


def read_file():
    name = 'Dataset/u.data'
    dataset = {}
    with open(name, newline='') as f:
        reader = csv.reader(f, delimiter = '\t')
        for row in reader:
            userid = int(row[0])
            itemid = int(row[1])
            rating = float(row[2])
            if userid not in dataset:
                dataset[userid] = {}
            dataset[userid][itemid] = rating
    return dataset


def get_age_bracket(age):
    if 7 <= age <= 14:
        return 23
    if 14 <= age <= 21:
        return 24
    if 22 <= age <= 28:
        return 25
    if 29 <= age <= 36:
        return 26
    if 37 <= age <= 48:
        return 27
    if 49 <= age <= 55:
        return 28
    if 56 <= age <= 65:
        return 29
    return 30


def get_occupation(occ):
    occupation = ['administrator',
                  'artist',
                  'doctor',
                  'educator',
                  'engineer',
                  'entertainment',
                  'executive',
                  'healthcare',
                  'homemaker',
                  'lawyer',
                  'librarian',
                  'marketing',
                  'none',
                  'other',
                  'programmer',
                  'retired',
                  'salesman',
                  'scientist',
                  'student',
                  'technician',
                  'writer']
    for i in range(len(occupation)):
        if occupation[i] == occ:
            return i + 2
    return 0


# The users are represented by binary vector of size 31 (gender-2, occupation-21 and age-8).
def user_features():
    name = 'Dataset/u.user'
    user_vector = {}
    with open(name, encoding="ISO-8859-1", newline='') as f:
        reader = csv.reader(f, delimiter='|')
        for row in reader:
            id = int(row[0])
            user_vector[id] = np.zeros(31)
            if row[2] == 'M':
                user_vector[id][0] = 1
            else:
                user_vector[id][1] = 1
            user_vector[id][get_occupation(row[3])] = 1
            user_vector[id][get_age_bracket(int(row[1]))] = 1
    return user_vector


def main():
    accuracy = 0
    global item_feature, user_feature
    item_feature, user_feature = item_features(), user_features()
    data = np.random.permutation(user_count + 1)
    b = np.array([0])
    data = np.setdiff1d(data, b)
    global rating_dataset
    rating_dataset = read_file()
    k_fold(data)


def k_fold(dataset):
    error = 0.0
    k = int(len(dataset) / 5)

    start = time.time()
    print("Iteration 1")
    test = dataset[:k]
    train = dataset[k:]
    temp = model(test, train)
    error += temp
    print(temp)

    print("Iteration 2")
    test = dataset[k:2 * k]
    train = np.concatenate((dataset[:k], dataset[2 * k:]))
    temp = model(test, train)
    error += temp
    print(temp)

    print("Iteration 3")
    test = dataset[2 * k:3 * k]
    train = np.concatenate((dataset[:2 * k], dataset[3 * k:]))
    temp = model(test, train)
    error += temp
    print(temp)

    print("Iteration 4")
    test = dataset[3 * k:4 * k]
    train = np.concatenate((dataset[:3 * k], dataset[4 * k:]))
    temp = model(test, train)
    error += temp
    print(temp)

    print("Iteration 5")
    test = dataset[4 * k:]
    train = dataset[:4 * k]
    temp = model(test, train)
    error += temp
    print(temp)

    end = time.time()
    print("Time taken = ", end - start)
    print(error / 5)

    return error / 5


def model(test, train):
    train_features = []
    train_ratings = []
    test_features = []
    test_ratings = []
    for i in train:
        for j in rating_dataset[i].keys():
            temp = np.concatenate((user_feature[i], item_feature[j]))
            train_features.append(temp)
            train_ratings.append(rating_dataset[i][j])
    for i in test:
        for j in rating_dataset[i].keys():
            test_features.append(np.concatenate((user_feature[i], item_feature[j])))
            test_ratings.append(rating_dataset[i][j])

    print("Going to predict now")
    clf = svm.SVC(gamma=0.001, C=100.)
    clf.fit(train_features, train_ratings)
    clf.predict(test_features)
    # print( len(train_features), len(train_ratings), len(test_features), len(test_ratings) )
    return 0

main()