# -*- coding: utf-8 -*-

# Python standard library
import random

# Numpy
import numpy as np

# Pandas
import pandas as pd

# Torch
import torch
from torch.utils.data import Dataset, DataLoader

# Colab
#from google.colab import drive
#drive.mount('/content/drive')

# Train dataset is {train_positive} Union {train_negative}
class TrainDataset(Dataset):
    def __init__(self, table):
        self.users_list, self.items_list, self.relevance_list = [], [], []
        
        # For each user
        for _, user_id, _, train_positive, _, train_negative, _, _ in table.itertuples():

            # Add positive train interactions
            for item in train_positive:
                self.users_list.append(user_id)
                self.items_list.append(item)
                self.relevance_list.append(1.0)

            # Add negative train interactions
            for item in train_negative:
                self.users_list.append(user_id)
                self.items_list.append(item)
                self.relevance_list.append(0.0)

    def __len__(self):
        return len(self.users_list)

    def __getitem__(self, idx):
        user_id = torch.tensor(self.users_list[idx], dtype = torch.long)
        item_id = torch.tensor(self.items_list[idx], dtype = torch.long)
        relevance = torch.tensor(self.relevance_list[idx], dtype = torch.float)

        return user_id, item_id, relevance

# Validation dataset is {valid_positive} Union {valid_negative}
class ValidationDataset(Dataset):
    def __init__(self, table):
        self.users_list, self.items_list, self.relevance_list = [], [], []
        
        # For each user
        for _, user_id, _, _, _, _, valid_positive, valid_negative in table.itertuples():

            # Add positive validation interactions
            for item in valid_positive:
                self.users_list.append(user_id)
                self.items_list.append(item)
                self.relevance_list.append(1.0)

            # Add negative validation interactions
            for item in valid_negative:
                self.users_list.append(user_id)
                self.items_list.append(item)
                self.relevance_list.append(0.0)

    def __len__(self):
        return len(self.users_list)

    def __getitem__(self, idx):
        user_id = torch.tensor(self.users_list[idx], dtype = torch.long)
        item_id = torch.tensor(self.items_list[idx], dtype = torch.long)
        relevance = torch.tensor(self.relevance_list[idx], dtype = torch.float)

        return user_id, item_id, relevance

# Complete train dataset is {train_positive} Union {train_negative} Union {valid_positive} Union {valid_negative}
class CompleteTrainDataset(Dataset):
    def __init__(self, train_dataset, valid_dataset):
        self.users_list = []
        self.users_list.extend(train_dataset.users_list)
        self.users_list.extend(valid_dataset.users_list)

        self.items_list = []
        self.items_list.extend(train_dataset.items_list)
        self.items_list.extend(valid_dataset.items_list)

        self.relevance_list = []
        self.relevance_list.extend(train_dataset.relevance_list)
        self.relevance_list.extend(valid_dataset.relevance_list)

        self.len = len(train_dataset) + len(valid_dataset)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        user_id = torch.tensor(self.users_list[idx], dtype = torch.long)
        item_id = torch.tensor(self.items_list[idx], dtype = torch.long)
        relevance = torch.tensor(self.relevance_list[idx], dtype = torch.float)

        return user_id, item_id, relevance

# Test dataset is {test_positive} Union {100 random samples from unknown}
class TestDataset(Dataset):
    def __init__(self, table, num_unknown=100):
        self.num_unknown = num_unknown
        self.users_list, self.items_list, self.relevance_list = [], [], []
        
        # For each user
        for _, user_id, test_positive, _, unknown, _, _, _ in table.itertuples():

            # Add positive test interactions
            for item in test_positive:
                self.users_list.append(user_id)
                self.items_list.append(item)
                self.relevance_list.append(1.0)

            # Add sampled unknown interactions
            for item in random.sample(list(unknown), self.num_unknown):
                self.users_list.append(user_id)
                self.items_list.append(item)
                self.relevance_list.append(np.nan)

    def __len__(self):
        return len(self.users_list)

    def __getitem__(self, idx):
        user_id = torch.tensor(self.users_list[idx], dtype = torch.long)
        item_id = torch.tensor(self.items_list[idx], dtype = torch.long)
        relevance = torch.tensor(self.relevance_list[idx], dtype = torch.float)

        return user_id, item_id, relevance

def data_loaders_from_data(path, BATCH_SIZE):
    ratings = pd.read_csv(path,
                        sep = '\t',
                        names = ['user_id', 'item_id', 'rating', 'timestamp'],
                        )
    # THEORETICAL FRAMEWORK
    #
    # The dataset contains ratings of items by users
    #
    # For each couple (u, i) of a user and a item, we consider
    # the discrete random variable R(u, i) = "item i is relevant for user u"
    # R(u, i) € {0, 1} for every u, i
    # In this scenario "relevant" doesn't assume the usual meaning
    # it refers to a "relevance" by a reccomender system point of view
    # 
    # If a user u rated an item i, then he interacted with it
    # in this case we assume to have a realization r(u, i) = 1 of R(u, i)
    #
    # Since we only have items ratings in our dataset, we don't have any "negative"
    # realization of R(u, i), i.e. any realization r(u, i) = 0
    #
    # For this reason we make a sampling from the unrealized variables, i.e
    # the variables R(u, i) such in our dataset there is not a rating of item i by user u
    #
    # Given our final dataset, consisting in realizations of a subset of the variables {R(u, i): u user, i item}
    # we want to maximize the likelihood function of the model, i.e.
    # L = product of p(R(u, i) = r(u, i))
    #   = (product of p(R(u, i) = 1)) x (product of p(R(u, i) = 0)   splitting the products between positive and negative realizations
    #
    # Our model returns the value f(u, i) = p(R(u, i) = 1) i.e. the "probability of relevance of item i for user u"
    # Passing to maximize the log likelihood function we want to maximize:
    # log L = (sum of log p(R(u, i) = 1)) + (sum of log p(R(u,i) = 0))
    #       = (sum of log p(R(u, i) = 1)) + (sum of log (1 - p(R(u, i) = 1)))
    #       = (sum of log f(u, i)) + (sum of log (1 - f(u, i)))
    #       = BCE of f

    num_users = ratings['user_id'].nunique()
    num_items = ratings['item_id'].nunique()
    known_positive_size_list = list(ratings.groupby('user_id').size())

    # I want user indexing to start from 0
    ratings['user_id'] = ratings['user_id'] - 1

    # I want item indexing to start from 0
    ratings['item_id'] = ratings['item_id'] - 1

    # For each user consider the set of items of which we know he interacted with
    relevance_table = ratings.groupby('user_id')['item_id'].apply(set).reset_index().rename(columns = {'item_id': 'known_positive'})

    # Set of all items
    items_set = set(range(num_items))

    # For each user consider the set of items of which we have NO interaction information
    relevance_table['unknown'] = relevance_table['known_positive'].apply(lambda x: items_set - x)

    # We don't know any negative interaction
    # We create them by sampling:

    # For each user
    #     Random sample items for each known_positive
    #     Assume that the probability of interaction for those items with the user is 0.0

    # How many negative interactions to add for each user
    def num_ng(user_id):
        known_positive_size = known_positive_size_list[user_id]
        if known_positive_size <= 50:
            return known_positive_size * 4
        if known_positive_size <= 100:
            return known_positive_size * 2
        if known_positive_size <= 200:
            return known_positive_size
        return 10

    relevance_table['known_negative'] = relevance_table.apply(lambda row: set(random.sample(list(row['unknown']), num_ng(row['user_id']))), axis = 'columns')
    relevance_table['unknown']  = relevance_table['unknown'] - relevance_table['known_negative']

    # Put aside the last known positive interactions for testing models(i.e. leave-one-out evaluation method)
    # How to do it:

    #     Add column ranking how old is the interaction
    ratings['oldness'] = ratings['timestamp'].groupby(ratings['user_id']).rank(method = 'first', ascending = False)

    #     Get the most recent interaction for each user
    most_recent_interactions = ratings.groupby('oldness').get_group(1.0)

    #     Create column to add to relevance_table
    last_known_positive = most_recent_interactions.groupby('user_id')['item_id'].apply(set).reset_index().rename(columns = {'item_id': 'last_known_positive'})

    #     Add the column
    relevance_table = pd.merge(last_known_positive, relevance_table, on = 'user_id')
    relevance_table['known_positive'] = relevance_table['known_positive'] - relevance_table['last_known_positive']

    # Rename some columns
    relevance_table.rename(columns = {'last_known_positive' : 'test_positive', 'known_positive': 'train_positive', 'known_negative': 'train_negative'}, inplace='True')

    # Validation data
    relevance_table['validation'] = relevance_table.apply(lambda row: set(random.sample(list(row['train_positive'] | row['train_negative']), 1)), axis = 'columns')

    relevance_table['valid_positive'] = relevance_table.apply(lambda row: row['validation'] & row['train_positive'], axis = 'columns')
    relevance_table['valid_negative'] = relevance_table.apply(lambda row: row['validation'] & row['train_negative'], axis = 'columns')

    relevance_table['train_positive']  = relevance_table['train_positive'] - relevance_table['validation']
    relevance_table['train_negative']  = relevance_table['train_negative'] - relevance_table['validation']

    relevance_table.drop(columns = 'validation', inplace = True)

    # print(relevance_table)

    train_dataset = TrainDataset(relevance_table)
    validation_dataset = ValidationDataset(relevance_table)
    
    train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True)
    valid_loader = DataLoader(validation_dataset, batch_size = BATCH_SIZE, shuffle = True)
    complete_train_loader = DataLoader(CompleteTrainDataset(train_dataset, validation_dataset), batch_size = BATCH_SIZE, shuffle = True)
    test_loader = DataLoader(TestDataset(relevance_table), batch_size = 101, shuffle = False)

    return train_loader, valid_loader, complete_train_loader, test_loader