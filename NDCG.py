from abc import abstractmethod
import numpy as np
import math
import collections
import sys

RANK_MAX = 10

class RelevanceEstimator():

    def __init__(self, true_relevances, minimum_occurences = 10):
        # relevances should be a dict with structure relevances[qid][uid] -> relevance
        self.relevances = true_relevances
        self.minimum_occurences = minimum_occurences

    def _group_sessions_if_useful(self, sessions, useful):
        """
         Group sessions based on query
        """
        sessions_dict = dict()
        for session in sessions:
            if session.query in useful:
                if not session.query in sessions_dict:
                    sessions_dict[session.query] = []
                sessions_dict[session.query].append(session)
        return sessions_dict

    def evaluate(self, click_model, search_sessions, k):
        """
         Return the NDCG@k of the rankings given by the model for the given sessions.
        """
        # Only use queries that occur more than MINUMUM_OCCURENCES times and have a true relevance
        counter = collections.Counter([session.query for session in search_sessions])
        useful_sessions = [query_id for query_id in counter if counter[query_id] >= self.minimum_occurences and query_id in self.relevances]
        print '\ttotal unique sessions: %d' % len(counter)
        print '\tuseful sessions: %d' % len(useful_sessions)

        # Group sessions by query
        sessions_dict = self._group_sessions_if_useful(search_sessions, useful_sessions)
        total_ndcg = 0
        not_useful = 0
        total_query = 0

        # For every useful query get the predicted relevance and compute NDCG
        for query_id in useful_sessions:
            
            rel = self.relevances[query_id]
            ideal_ranking = sorted(rel.values(),reverse = True)[:k]
            
            # Only use query if there is a document with a positive ranking. (Otherwise IDCG will be 0 -> NDCG undetermined.)
            if not any(ideal_ranking):
                not_useful += 1
                continue
            
            current_sessions = sessions_dict[query_id]
            pred_rels = dict()
            for session in current_sessions:
                total_query += 1
                for rank, result in enumerate(session.web_results):
                    if not result.id in pred_rels:
                        pred_rels[result.id] = click_model.predict_relevance(session.query, result.id)
            ranking = sorted([doc for doc in pred_rels],key = lambda doc : pred_rels[doc], reverse = True)
            
            ranking_relevances = self.get_relevances(query_id, ranking[:k])
            
            dcg = self.dcg(ranking_relevances)
            idcg = self.dcg(ideal_ranking)            
            ndcg = dcg / idcg
            total_ndcg += ndcg

        # If too few queries, there might not be any useful queries that also have a ranking in the true_relevances.
        assert not len(useful_sessions)-not_useful is 0
        assert total_query + not_useful == len(search_sessions)

         # Average NDCG over all queries
        return total_ndcg / (len(useful_sessions)-not_useful)


    #TODO: If not found say 0.5. Probably not correct. Otherwise no evaluations in small testset..
    def get_relevances(self, query_id, ranking):
        ranking_relevances = []
        for doc in ranking:
            if doc in self.relevances[query_id]:
                ranking_relevances.append(self.relevances[query_id][doc])
            else:
                ranking_relevances.append(0.5)

        return ranking_relevances

        

    def dcg(self, ranking):
        """
            Computes the DCG for a given ranking.
        """
        return sum([(2**r-1)/math.log(i+2,2) for i,r in enumerate(ranking)])
