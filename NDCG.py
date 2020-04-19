from abc import abstractmethod
import numpy as np
import math
import collections
import sys

RANK_MAX = 10

class RelevanceEstimator():

    def __init__(self, true_relevances, minimum_occurences = 10):
        # relevances should be a dict with structure relevances[qid][uid] -> relevance
        # self.logger = logging.getLogger("NCM")
        self.relevances = true_relevances
        self.minimum_occurences = minimum_occurences

    def _group_sids_if_useful(self, relevance_queries, useful_qids):
        """
         Group sids based on query
        """
        qid_queries = dict()
        for info_per_query in relevance_queries:
            if info_per_query['qid'] in useful_qids:
                if not info_per_query['qid'] in qid_queries:
                    qid_queries[info_per_query['qid']] = []
                qid_queries[info_per_query['qid']].append(info_per_query)
        return qid_queries

    def evaluate(self, model, relevance_queries, k):
        """
         Return the NDCG@k of the rankings given by the model for the given sessions.
        """
        # Only use queries that occur more than MINUMUM_OCCURENCES times and have a true relevance
        unique_qid_counter = collections.Counter([info_per_query['qid'] for info_per_query in relevance_queries])
        useful_qids = [qid for qid in unique_qid_counter if unique_qid_counter[qid] >= self.minimum_occurences and qid in self.relevances]
        #self.logger.info('total unique relevance queries: %d' % len(unique_qid_counter))
        #self.logger.info('useful unique relevance queries: %d' % len(useful_qids))

        # Group sessions by query
        qid_queries = self._group_sids_if_useful(relevance_queries, useful_qids)
        total_ndcg = 0
        not_useful = 0
        total_query = 0

        # For every useful query get the predicted relevance and compute NDCG
        for qid in useful_qids:
            
            rel = self.relevances[qid]
            ideal_ranking_relevances = sorted(rel.values(), reverse = True)[:k]
            
            # Only use query if there is a document with a positive ranking. (Otherwise IDCG will be 0 -> NDCG undetermined.)
            if not any(ideal_ranking_relevances):
                not_useful += 1
                continue
            
            queries = qid_queries[qid]
            pred_rels = dict()
            for info_per_query in queries:
                total_query += 1
                for rank, uid in enumerate(info_per_query['uids']):
                    if not uid in pred_rels:
                        pred_rels[uid] = model.predict_relevance(info_per_query['qid'], uid)
            ranking = sorted([uid for uid in pred_rels], key = lambda uid : pred_rels[uid], reverse = True)
            ranking_relevances = self.get_relevances(qid, ranking[:k])
            
            dcg = self.dcg(ranking_relevances)
            idcg = self.dcg(ideal_ranking_relevances)            
            ndcg = dcg / idcg
            total_ndcg += ndcg

        # If too few queries, there might not be any useful queries that also have a ranking in the true_relevances.
        assert not len(useful_qids) - not_useful == 0
        assert total_query + not_useful == len(relevance_queries)

         # Average NDCG over all queries
        return total_ndcg / (len(useful_qids) - not_useful)

    def get_relevances(self, qid, ranking):
        '''
         return the corresponding true relevance value from the uid ranking list
        '''
        ranking_relevances = []
        for uid in ranking:
            if uid in self.relevances[qid]:
                ranking_relevances.append(self.relevances[qid][uid])
            else:
                # if the uid is not in self.relevances[qid], return 0.5, which means that do not know whether relevant or not
                ranking_relevances.append(0.5)
        return ranking_relevances

    def dcg(self, ranking_relevances):
        """
         Computes the DCG for a given ranking_relevances
        """
        return sum([(2 ** relevance - 1) / math.log(rank + 2, 2) for rank , relevance in enumerate(ranking_relevances)])
