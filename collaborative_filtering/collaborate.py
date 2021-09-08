import time
import numpy as np
from collaborative_filtering.similarities import person_sim as sim
from sys import maxsize

INT_MIN = -maxsize - 1

class Collaborate:
    """
    Contains methods to perform collabarative filtering with and without baseline approach.
    """
    def __init__(self, M):
        """
        Initialize utility  matrix
        Input:
        M: Input Matrix
        """
        self.M = M

    def estimate(self, user, item, k=2, baseline=False):
        """
        For a given input user and item, this function estimates rating

        Input:
        user (int): Index of User
        item (int): Index of Item
        k (int): Nearest neighbours taken based on similarity (default = 2)
        baseline (bool): Toggle baseline offset (default = False)

        """
        # Ratings matrix
        r = self.M
        # Mean baseline deviation
        mu = 0
        # User baseline deviation
        baseline_user = 0
        # Item baseline deviation
        baseline_item = 0
        # With baseline deviation considered
        if baseline is True:
            mu = sum(sum(r))/np.count_nonzero(r)
            baseline_user = sum(r[:, user])/np.count_nonzero(r[:, user]) - mu
            baseline_item = sum(r[item])/np.count_nonzero(r[item]) - mu
        # Overall baseline deviation
        b = mu + baseline_user + baseline_item
        ### Rating estimation ###
        # Calculate similarities
        S = np.zeros(r.shape[0])
        for i in range(r.shape[0]):
            S[i] = sim(r, item, i)
            if np.isnan(S[i]):
                S[i] = 0
        # Estimate the rating
        num = 0
        s_list = list(S)
        max = sorted(S, reverse=True)[1:3]

        max_id = list()
        for i in max:
            if i != 1:
                if len(max_id) == k:
                    break
                else:
                    max_id.append(s_list.index(i))

        den = np.sum(max)
        den = np.sum(max)

        for i in max_id:
            if baseline:
                b_ui = baseline_user + sum(r[i])/np.count_nonzero(r[i])
            else:
                b_ui = 0
            num += (r[i, user] - b_ui)*S[i]

        rating = b + (num/den)
        if np.isnan(rating):
            rating = 0

        return rating

    def fill(self, k=2, baseline=False):
        """
        Fills gaps in utility matrix using CF estimates

        Input:
        k : Nearest neighbours taken (default = 2)
        baseline : Toggles baseline offset (default = False)
        """
        fill_gap = np.zeros(self.M.shape)
        for i in range(self.M.shape[0]):
            for j in range(self.M.shape[1]):
                if self.M[i, j] == 0:
                    fill_gap[i, j] = self.estimate(j, i, k=k, baseline=baseline)
                else:
                    fill_gap[i, j] = self.M[i, j]
        return fill_gap
