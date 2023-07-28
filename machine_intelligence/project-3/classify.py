#!/usr/bin/env python3

import json
import argparse
import os
import numpy as np
import tqdm
import random

# Supress sklearn warnings
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# seed value
# (ensures consistent dataset splitting between runs)
SEED = 0


def parse_args():
    """
    Parse arguments.
    """
    parser = argparse.ArgumentParser()

    def check_path(parser, x):
        if not os.path.exists(x):
            parser.error("That directory {} does not exist!".format(x))
        else:
            return x
    parser.add_argument('-r', '--root', type=lambda x: check_path(parser, x), 
                        help='The path to the root directory containing feature files.')
    parser.add_argument('-s', '--split', type=float, default=0.7, 
                        help='The percentage of samples to use for training.')

    return parser.parse_args()


def load_data(root, min_samples=20, max_samples=1000):
    """Load json feature files produced from feature extraction.
    The device label (MAC) is identified from the directory in which the feature file was found.
    Returns X and Y as separate multidimensional arrays.
    The instances in X contain only the first 6 features.
    The ports, domain, and cipher features are stored in separate arrays for easier process in stage 0.
    Parameters
    ----------
    root : str
           Path to the directory containing samples.
    min_samples : int
                  The number of samples each class must have at minimum (else it is pruned).
    max_samples : int
                  Stop loading samples for a class when this number is reached.
    Returns
    -------
    features_misc : numpy array
    features_ports : numpy array
    features_domains : numpy array
    features_ciphers : numpy array
    labels : numpy array
    """
    X = []
    X_p = []
    X_d = []
    X_c = []
    Y = []

    port_dict = dict()
    domain_set = set()
    cipher_set = set()

    # create paths and do instance count filtering
    fpaths = []
    fcounts = dict()
    for rt, dirs, files in os.walk(root):
        for fname in files:
            path = os.path.join(rt, fname)
            label = os.path.basename(os.path.dirname(path))
            name = os.path.basename(path)
            if name.startswith("features") and name.endswith(".json"):
                fpaths.append((path, label, name))
                fcounts[label] = 1 + fcounts.get(label, 0)

    # load samples
    processed_counts = {label:0 for label in fcounts.keys()}
    for fpath in tqdm.tqdm(fpaths):
        path = fpath[0]
        label = fpath[1]
        if fcounts[label] < min_samples:
            continue
        if processed_counts[label] >= max_samples:
            continue
        processed_counts[label] += 1
        with open(path, "r") as fp:
            features = json.load(fp)
            instance = [features["flow_volume"],
                        features["flow_duration"],
                        features["flow_rate"],
                        features["sleep_time"],
                        features["dns_interval"],
                        features["ntp_interval"]]
            X.append(instance)
            X_p.append(list(features["ports"]))
            X_d.append(list(features["domains"]))
            X_c.append(list(features["ciphers"]))
            Y.append(label)
            domain_set.update(list(features["domains"]))
            cipher_set.update(list(features["ciphers"]))
            for port in set(features["ports"]):
                port_dict[port] = 1 + port_dict.get(port, 0)

    # prune rarely seen ports
    port_set = set()
    for port in port_dict.keys():
        if port_dict[port] > 10:
            port_set.add(port)

    # map to wordbag
    print("Generating wordbags ... ")
    for i in tqdm.tqdm(range(len(Y))):
        X_p[i] = list(map(lambda x: X_p[i].count(x), port_set))
        X_d[i] = list(map(lambda x: X_d[i].count(x), domain_set))
        X_c[i] = list(map(lambda x: X_c[i].count(x), cipher_set))

    return np.array(X).astype(float), np.array(X_p), np.array(X_d), np.array(X_c), np.array(Y)


def classify_bayes(X_tr, Y_tr, X_ts, Y_ts):
    """
    Use a multinomial naive bayes classifier to analyze the 'bag of words' seen in the ports/domain/ciphers features.
    Returns the prediction results for the training and testing datasets as an array of tuples in which each row
      represents a data instance and each tuple is composed as the predicted class and the confidence of prediction.
    Parameters
    ----------
    X_tr : numpy array
           Array containing training samples.
    Y_tr : numpy array
           Array containing training labels.
    X_ts : numpy array
           Array containing testing samples.
    Y_ts : numpy array
           Array containing testing labels
    Returns
    -------
    C_tr : numpy array
           Prediction results for training samples.
    C_ts : numpy array
           Prediction results for testing samples.
    """
    classifier = MultinomialNB()
    classifier.fit(X_tr, Y_tr)

    # produce class and confidence for training samples
    C_tr = classifier.predict_proba(X_tr)
    C_tr = [(np.argmax(instance), max(instance)) for instance in C_tr]

    # produce class and confidence for testing samples
    C_ts = classifier.predict_proba(X_ts)
    C_ts = [(np.argmax(instance), max(instance)) for instance in C_ts]

    return C_tr, C_ts


def do_stage_0(Xp_tr, Xp_ts, Xd_tr, Xd_ts, Xc_tr, Xc_ts, Y_tr, Y_ts):
    """
    Perform stage 0 of the classification procedure:
        process each multinomial feature using naive bayes
        return the class prediction and confidence score for each instance feature
    Parameters
    ----------
    Xp_tr : numpy array
           Array containing training (port) samples.
    Xp_ts : numpy array
           Array containing testing (port) samples.
    Xd_tr : numpy array
           Array containing training (domain) samples.
    Xd_ts : numpy array
           Array containing testing (domain) samples.
    Xc_tr : numpy array
           Array containing training (cipher) samples.
    Xc_ts : numpy array
           Array containing testing (cipher) samples.
    Y_tr : numpy array
           Array containing training labels.
    Y_ts : numpy array
           Array containing testing labels
    Returns
    -------
    resp_tr : numpy array
              Prediction results for training (port) samples.
    resp_ts : numpy array
              Prediction results for testing (port) samples.
    resd_tr : numpy array
              Prediction results for training (domains) samples.
    resd_ts : numpy array
              Prediction results for testing (domains) samples.
    resc_tr : numpy array
              Prediction results for training (cipher suites) samples.
    resc_ts : numpy array
              Prediction results for testing (cipher suites) samples.
    """
    # perform multinomial classification on bag of ports
    resp_tr, resp_ts = classify_bayes(Xp_tr, Y_tr, Xp_ts, Y_ts)

    # perform multinomial classification on domain names
    resd_tr, resd_ts = classify_bayes(Xd_tr, Y_tr, Xd_ts, Y_ts)

    # perform multinomial classification on cipher suites
    resc_tr, resc_ts = classify_bayes(Xc_tr, Y_tr, Xc_ts, Y_ts)

    return resp_tr, resp_ts, resd_tr, resd_ts, resc_tr, resc_ts


def do_stage_1(X_tr, X_ts, Y_tr, Y_ts):
    """
    Perform stage 1 of the classification procedure:
        train a random forest classifier using the NB prediction probabilities
    Parameters
    ----------
    X_tr : numpy array
           Array containing training samples.
    Y_tr : numpy array
           Array containing training labels.
    X_ts : numpy array
           Array containing testing samples.
    Y_ts : numpy array
           Array containing testing labels
    Returns
    -------
    pred : numpy array
           Final predictions on testing dataset.
    """
    model = RandomForestClassifier(n_jobs=-1, n_estimators=1, oob_score=True)
    model.fit(X_tr, Y_tr)

    score = model.score(X_ts, Y_ts)
    print("RF accuracy = {}".format(score))

    pred = model.predict(X_ts)
    return pred


### count samples per class
counts = lambda x: np.unique(x, return_counts=True)[1]


def gini_impurity(left, right):
    """Calculate Gini Impurity given the data has been split into two parts
        input: left and right splits at a node
        return: gini impurity score for the node
    """
    len_total = len(left) + len(right)
    gini_impurity = 0
    for part in [left, right]:
        len_part = len(part)
        ### Calculate Gini impurity per part 
        part_impurity = 1
        for cc in counts(part):
            part_impurity -= (cc/len_part)**2
        gini_impurity += part_impurity * len_part/len_total
    return gini_impurity


def best_split_loc(x, y):
    """Find Best split location by checking each feature 
        point in the samples in the samples to be split
        input: x ---> features
               y ---> class labels
        return: best_feature2_split ---> best feature index
                best_split ---> best split value in the feature 
                best_gini ---> best gini impurity
                Rest names are self explanatory
    """
    best_gini = gini_impurity(y, y) ### initial value to start with
    best_feature2_split = -1
    best_split = -1
    best_left_x = -1
    best_left_y = -1
    best_right_x = -1
    best_right_y = -1
    # print(best_gini)
    for col_idx in range(x.shape[1]):
        ### Find unique values in the column
        unique_values = np.unique(x[:, col_idx])
        ### Find mean split positions
        splits = (unique_values[:-1] + unique_values[1:])/2
        ### For all possible splits
        for split in splits:
            ### Left part of split
            left_map = x[:, col_idx] <= split
            left_x, left_y = x[left_map], y[left_map]
            ### Right part of split
            right_map = x[:, col_idx] > split
            right_x, right_y = x[right_map], y[right_map]
            ### gini impurity for the split configuration
            gini = gini_impurity(left_y, right_y)
            ### select the configuration with least gini impurity
            if gini < best_gini:
                best_gini = gini
                best_feature2_split = col_idx
                best_split = split
                best_left_x = left_x
                best_left_y = left_y
                best_right_x = right_x
                best_right_y = right_y
    return (best_feature2_split, best_split, best_gini, 
            best_left_x, best_left_y, best_right_x, best_right_y)

class node_choice():
    """
    Choice on a node point: left or right based upon 
    best column to split on with optimal splitting value.
    """
    def __init__(self, left=None, right=None, col_idx=None, split_value=None, label=None):
        self.left = left
        self.right = right
        self.label = label
        self.part = lambda x: x[col_idx] < split_value ### choice function
        
    def left_or_right(self, x):
        if self.label is not None: ### For leaf node
            return self.label
        elif self.part(x): ### For child node left
            #print('left')
            return self.left.left_or_right(x)
        else: ### For child node right
            #print('right')
            return self.right.left_or_right(x)


class decision_tree():
    def __init__(self, max_depth, min_node):
        self.tree = None
        self.max_depth = max_depth
        self.min_node = min_node
    
    def fit(self, x, y):
        self.tree = self.construct(x, y)
        
    def mode(self, x):
        return np.unique(x)[np.argmax(counts(x))]
        
    def construct(self, x, y, branch_depth=5):
        if len(y)==0:
            print('length zero')
            return None
        elif len(np.unique(y))==1:
            print('unique 1')
            print(y)
            print(y.shape)
            return node_choice(label=y[0])
        else:
            # print("here")
            gini_parent = gini_impurity(y, y)
            col_idx, split_value, gini, left_x, left_y, right_x, right_y = best_split_loc(x, y)
            #print('gini_parent: {}'.format(gini_parent))
            #print('gini: {}'.format(gini))
            #print('branch_depth: {}'.format(branch_depth))
            if branch_depth >= self.max_depth or len(y) < self.min_node or gini >= gini_parent:
                #print('mode')
                return node_choice(label=self.mode(y))
            else:
                branch_depth += 1
                left = self.construct(left_x, left_y, branch_depth)
                right = self.construct(right_x, right_y, branch_depth)
                return node_choice(left, right, col_idx, split_value)
    
    def pred(self, x):
        predictions = []
        for row in x:
            label = self.tree.left_or_right(row)
            predictions.append(label)
        return predictions


def custom_do_stage_1(X_tr, X_ts, Y_tr, Y_ts):
    """
    Perform stage 1 of the classification procedure:
        train a random forest classifier using the NB prediction probabilities
    Parameters
    ----------
    X_tr : numpy array
           Array containing training samples.
    Y_tr : numpy array
           Array containing training labels.
    X_ts : numpy array
           Array containing testing samples.
    Y_ts : numpy array
           Array containing testing labels
    Returns
    -------
    pred : numpy array
           Final predictions on testing dataset.
    """
    
    ### maximum depth of the tree
    max_depth = 15
    
    ###  minimum number of samples allowed per leaf node
    min_node = 1

    ### number of trees
    n_trees = 20

    ### The percentage of the dataset, data_frac, to use when building each tree
    data_frac = 0.95

    ### percantage of the features to preserve
    per = 0.85

    ### labels
    labels = 24

    sub_train_len = int(data_frac * len(X_tr))
    num_feat = int(per * X_tr.shape[1])
    preds = []
    
    for tree in range(n_trees):
        # create subset of the train data with random sampling with replacement
        mask_data = random.choices(range(len(X_tr)), k=sub_train_len)
        mask_features = random.choices(range(X_tr.shape[1]), k=num_feat)
        train_sub_X = X_tr[np.ix_(mask_data, mask_features)]
        train_sub_Y = Y_tr[mask_data]
        test_sub_X = X_ts[:, mask_features]

        model = decision_tree(max_depth, min_node)
        model.fit(train_sub_X, train_sub_Y)
        prediction = model.pred(test_sub_X)
        preds.append(prediction)
        
    preds = np.array(preds)
    results =  np.full((X_ts.shape[0], labels), -1)

    for t_count in range(n_trees):
        for ind in range(X_ts.shape[0]):
            results[ind, preds[t_count,ind]] += 1

    final_results = []
    for res in range(X_ts.shape[0]):
        maxpos = np.argmax(results[res, :], ) 
        final_results.append(maxpos)

    return final_results


def main(args):
    """
    Perform main logic of program
    """
    # load dataset
    print("Loading dataset ... ")
    X, X_p, X_d, X_c, Y = load_data(args.root)

    # encode labels
    print("Encoding labels ... ")
    le = LabelEncoder()
    le.fit(Y)
    Y = le.transform(Y)

    print("Dataset statistics:")
    print("\t Classes: {}".format(len(le.classes_)))
    print("\t Samples: {}".format(len(Y)))
    print("\t Dimensions: ", X.shape, X_p.shape, X_d.shape, X_c.shape)

    # shuffle
    print("Shuffling dataset using seed {} ... ".format(SEED))
    s = np.arange(Y.shape[0])
    np.random.seed(SEED)
    np.random.shuffle(s)
    X, X_p, X_d, X_c, Y = X[s], X_p[s], X_d[s], X_c[s], Y[s]

    # split
    print("Splitting dataset using train:test ratio of {}:{} ... ".format(int(args.split*100), int((1-args.split)*100)))
    cut = int(len(Y) * args.split)
    X_tr, Xp_tr, Xd_tr, Xc_tr, Y_tr = X[:cut], X_p[:cut], X_d[:cut], X_c[:cut], Y[:cut]
    X_ts, Xp_ts, Xd_ts, Xc_ts, Y_ts = X[cut:], X_p[cut:], X_d[cut:], X_c[cut:], Y[cut:]

    # perform stage 0
    print("Performing Stage 0 classification ... ")
    p_tr, p_ts, d_tr, d_ts, c_tr, c_ts = \
        do_stage_0(Xp_tr, Xp_ts, Xd_tr, Xd_ts, Xc_tr, Xc_ts, Y_tr, Y_ts)

    # build stage 1 dataset using stage 0 results
    # NB predictions are concatenated to the quantitative attributes processed from the flows
    X_tr_full = np.hstack((X_tr, p_tr, d_tr, c_tr))
    X_ts_full = np.hstack((X_ts, p_ts, d_ts, c_ts))
    
    # print(X_tr_full)
    # # perform final classification
    print("Performing Stage 1 classification ... ")
    # # pred = do_stage_1(X_tr_full, X_ts_full, Y_tr, Y_ts)
    pred = custom_do_stage_1(X_tr_full, X_ts_full, Y_tr, Y_ts)
    # print classification report
    # print(len(pred))
    # print(len(Y_ts))
    print(classification_report(Y_ts, pred, target_names=le.classes_))


if __name__ == "__main__":
    # parse cmdline args
    args = parse_args()
    args.root= 'iot_data'
    main(args)
