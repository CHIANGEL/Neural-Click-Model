from abc import abstractmethod
import numpy as np
import math
import collections
import sys
import logging
import pprint
import logging

RANK_MAX = 10

class RelevanceEstimator():

    def __init__(self, minimum_occurrence = 10):
        self.logger = logging.getLogger("NCM")
        self.minimum_occurrence = minimum_occurrence

    def evaluate(self, model, dataset, relevance_queries, k):
        """
         Return the NDCG@k of the rankings given by the model for the given sessions.
        """
        # Only use queries that occur more than MINUMUM_OCCURENCES times
        unique_qid_counter = collections.Counter([info_per_query['qid'] for info_per_query in relevance_queries])
        useful_qids = [qid for qid in unique_qid_counter if unique_qid_counter[qid] >= self.minimum_occurrence]
        useful_relevance_queries = [info_per_query for info_per_query in relevance_queries if info_per_query['qid'] in useful_qids]
        #self.logger.info('total unique relevance queries: {}'.format(len(unique_qid_counter)))
        #self.logger.info('useful unique relevance queries: {}'.format(len(useful_qids)))
        #self.logger.info('total relevance queries after filering: {}'.format(len(useful_relevance_queries)))

        # For every useful query get the predicted relevance and compute NDCG
        total_ndcg = 0
        total_query = 0
        not_useful = 0
        for info_per_query in useful_relevance_queries:
            total_query += 1
            print('total_query: {}'.format(total_query))
            id = info_per_query['id']
            sid = info_per_query['sid']
            qid = info_per_query['qid']
            uids = info_per_query['uids']
            relevances = info_per_query['relevances']
            ideal_ranking_relevances = sorted(relevances, reverse=True)[:k]
            #print('  uids: {}'.format(uids))
            #print('  relevances: {}'.format(relevances))
            #print('  ideal_ranking_relevances: {}'.format(ideal_ranking_relevances))
            
            # Only use query if there is a document with a positive ranking. (Otherwise IDCG will be 0 -> NDCG undetermined.)
            if not any(ideal_ranking_relevances):
                pprint.pprint(info_per_query)
                not_useful += 1
                continue
            
            # Get the relevances computed by the model
            pred_rels = dict()
            for uid in uids:
                target_qid = dataset.query_qid[qid]
                target_uid = dataset.url_uid[uid]
                target_vid = dataset.uid_vid[dataset.url_uid[uid]]
                pred_rels[uid] = model.predict_relevance(target_qid, target_uid, target_vid).item()
            ranking = sorted([uid for uid in pred_rels], key = lambda uid : pred_rels[uid], reverse=True)
            ranking_relevances = [relevances[uids.index(uid)] for uid in ranking[:k]]
            #print('  pred_rels: {}'.format(pred_rels))
            #print('  ranking: {}'.format(ranking))
            #print('  ranking_relevances: {}'.format(ranking_relevances))
            
            #self.logger.info('{}: {}'.format('dcg', ranking_relevances))
            #self.logger.info('{}: {}'.format('idcg', ideal_ranking_relevances))
            dcg = self.dcg(ranking_relevances)
            idcg = self.dcg(ideal_ranking_relevances)
            if dcg > idcg:
                pprint.pprint(ranking_relevances)
                pprint.pprint(ideal_ranking_relevances)
                pprint.pprint(dcg)
                pprint.pprint(idcg)
                pprint.pprint(info_per_query)
                assert 0
            ndcg = dcg / idcg
            assert ndcg <= 1
            total_ndcg += ndcg

        # Checks
        assert total_query == len(useful_relevance_queries)
        assert len(useful_relevance_queries) - not_useful != 0

        # Average NDCG over all queries
        ndcg_version1 = total_ndcg / (len(useful_relevance_queries) - not_useful)
        ndcg_version2 = (total_ndcg + not_useful) / len(useful_relevance_queries)
        return ndcg_version1, ndcg_version2

    def dcg(self, ranking_relevances):
        """
         Computes the DCG for a given ranking_relevances
        """
        return sum([(2 ** relevance - 1) / math.log(rank + 2, 2) for rank, relevance in enumerate(ranking_relevances)])