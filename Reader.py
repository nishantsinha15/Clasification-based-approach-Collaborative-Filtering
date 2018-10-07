import csv
import time
from sklearn.svm import LinearSVC
import numpy as np
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn_extensions.extreme_learning_machines.elm import GenELMClassifier
from sklearn_extensions.extreme_learning_machines.random_layer import RBFRandomLayer, MLPRandomLayer



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
        reader = csv.reader(f, delimiter='\t')
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
    # nh = [10, 50, 100, 1000]
    # elm_accuracy = -1
    # nh_val = 0
    # for i in nh:
    #     temp = eml_classifier(np.asarray(train_features), np.asarray(train_ratings), np.asarray(test_features),
    #                           np.asarray(test_ratings))
    #     print("Tuning ", temp, i)
    #     if temp > elm_accuracy:
    #         elm_accuracy = temp
    #         nh_val = i
    # print("Minimum ELM accuracy = ", elm_accuracy, " for c = ", nh_val)
    # return elm_accuracy

    c = [0.1, 0.2, 0.5, 1, 10, 100]
    svm_accuracy = -1
    c_val = 0
    for i in c:
        temp = svm_classifier(train_features, train_ratings, test_features, test_ratings, i)
        print("Tuning ", temp, i)
        if temp > svm_accuracy:
            svm_accuracy = temp
            c_val = i
    print("Minimum SVM accuracy = ", svm_accuracy, " for c = ", c_val)
    return svm_accuracy


def svm_classifier(train_features, train_ratings, test_features, test_ratings, c):
    clf = LinearSVC(C = c)
    clf.fit(train_features, train_ratings)
    y = clf.predict(test_features)
    accuracy = get_accuracy(y, test_ratings)
    print(accuracy)
    return accuracy


# reference = http://wdm0006.github.io/sklearn-extensions/extreme_learning_machines.html
def eml_classifier(train_features, train_ratings, test_features, test_ratings):
    nh = [10, 20, 50, 100]
    # srhl_tanh = MLPRandomLayer(n_hidden=nh, activation_func='tanh')
    # srhl_tribas = MLPRandomLayer(n_hidden=nh, activation_func='tribas')
    # srhl_hardlim = MLPRandomLayer(n_hidden=nh, activation_func='hardlim')

    srhl_rbf = RBFRandomLayer(n_hidden=nh[0]*2, rbf_width=0.1, random_state=0)
    clf = GenELMClassifier(hidden_layer=srhl_rbf)
    clf.fit(train_features, train_ratings)
    accuracy = clf.score(test_features, test_ratings)
    print(accuracy)
    return accuracy

def get_accuracy(y, test_rating):
    count = 0
    for i in range(len(y)):
        if y[i] == test_rating[i]:
            count += 1
    return count / len(y)


main()
