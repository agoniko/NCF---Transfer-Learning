import random

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class Rating_Dataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        user, item, rating = self.data[idx]

        return (
            torch.tensor(user, dtype=torch.long),
            torch.tensor(item, dtype=torch.long),
            torch.tensor(rating, dtype=torch.float),
        )


# NCF_Data has to be called like this
# ml_100k = pd.read_csv(
# 	PATH,
# 	sep="\t",
# 	names = ['user_id', 'item_id', 'rating', 'timestamp'],
# 	engine='python')

# set the num_users, items
# num_users = ml_100k['user_id'].nunique()+1
# num_items = ml_100k['item_id'].nunique()+1
# construct the train and test datasets
# data = NCF_Data(ml_100k, args)


class NCF_Data:
    """
    Construct Dataset for NCF
    """

    def __init__(self, ratings: pd.DataFrame, args: dict):
        # self.ratings = ratings
        self.num_ng = args["num_ng"]
        self.num_ng_test = args["num_ng_test"]
        self.batch_size = args["batch_size"]
        ratings = ratings.copy(deep=True)
        ratings["rating"] = 1.0
        # self.preprocess_ratings = self._reindex(ratings)

        self.item_pool = set(ratings["item_id"].unique())

        self.train_ratings, self.test_ratings = self._leave_one_out(ratings)
        self.negatives = self._negative_sampling(ratings)

    def _reindex(self, ratings: pd.DataFrame) -> pd.DataFrame:
        """
        Process dataset to reindex userID and itemID, also set rating as binary feedback
        """
        user_list = ratings["user_id"].drop_duplicates()
        user2id = {w: i for i, w in enumerate(user_list)}

        item_list = ratings["item_id"].drop_duplicates()
        item2id = {w: i for i, w in enumerate(item_list)}

        ratings["user_id"] = ratings["user_id"].apply(lambda x: user2id[x])
        ratings["item_id"] = ratings["item_id"].apply(lambda x: item2id[x])
        ratings["rating"] = ratings["rating"].apply(lambda x: float(x > 0))
        return ratings

    def _leave_one_out(
        self, ratings: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        leave-one-out evaluation protocol in paper https://www.comp.nus.edu.sg/~xiangnan/papers/ncf.pdf
        """
        ratings["rank_latest"] = ratings.groupby(["user_id"])["timestamp"].rank(
            method="first", ascending=False
        )
        test = ratings.loc[ratings["rank_latest"] == 1]
        train = ratings.loc[ratings["rank_latest"] > 1]
        assert (
            train["user_id"].nunique() == test["user_id"].nunique()
        ), "Missing user in train data set"
        ratings.drop("rank_latest", axis=1, inplace=True)
        return (
            train[["user_id", "item_id", "rating"]],
            test[["user_id", "item_id", "rating"]],
        )

    def _negative_sampling(self, ratings: pd.DataFrame) -> pd.DataFrame:
        interact_status = (
            ratings.groupby("user_id")["item_id"]
            .apply(set)
            .reset_index()
            .rename(columns={"item_id": "interacted_items"})
        )
        interact_status["negative_items"] = interact_status["interacted_items"].apply(
            lambda x: self.item_pool - x
        )
        interact_status["negative_samples"] = interact_status["negative_items"].apply(
            lambda x: random.sample(x, self.num_ng_test)
        )
        return interact_status[["user_id", "negative_items", "negative_samples"]]

    def get_train_instance(self) -> DataLoader:
        data = []
        train_ratings = pd.merge(
            self.train_ratings,
            self.negatives[["user_id", "negative_items"]],
            on="user_id",
        )
        train_ratings["negatives"] = train_ratings["negative_items"].apply(
            lambda x: random.sample(x, self.num_ng)
        )
        for row in train_ratings.itertuples():
            data.append((int(row.user_id), int(row.item_id), float(row.rating)))
            for i in range(self.num_ng):
                data.append((int(row.user_id), int(i), 0.0))
        dataset = Rating_Dataset(data)
        return DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, num_workers=1
        )

    def get_test_instance(self) -> DataLoader:
        data = []
        test_ratings = pd.merge(
            self.test_ratings,
            self.negatives[["user_id", "negative_samples"]],
            on="user_id",
        )
        for row in test_ratings.itertuples():
            data.append((int(row.user_id), int(row.item_id), float(row.rating)))
            for i in row.negative_samples:
                data.append((int(row.user_id), int(i), 0.0))
        dataset = Rating_Dataset(data)
        return DataLoader(
            dataset, batch_size=self.num_ng_test + 1, shuffle=False, num_workers=1
        )
