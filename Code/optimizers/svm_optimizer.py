from classifiers.svm import SVMClassifier
import numpy as np

def generate_svm(feedbacks_to_apply, type_pos, type_neg):
    X, y = [], []

    found_neg, found_pos = False, False

    for feedback_to_apply in feedbacks_to_apply.values():
        vector, feedback = feedback_to_apply

        if feedback in type_pos:
            X.append(vector)
            y.append(1)
            found_pos = True
        elif feedback in type_neg:
            X.append(vector)
            y.append(-1)
            found_neg = True

    if found_neg and found_pos:
        svm = SVMClassifier(gamma=3.0)
        svm.fit(X, y)
        return svm
    else:
        return None


def get_new_candidates_by_svm(candidates, outer_svm, rel_svm, irr_svm, feedbacks):
    new_candidates = []

    for candidate in candidates:
        vector = np.array(candidate[0])
        img_id = candidate[1]
        dist = candidate[2]
        relevance = 1   # 0 indicates R+, 3 indicates I-

        if outer_svm is not None:
            pred_class = outer_svm.predict([vector])[0]

            if pred_class == 1:
                relevance = 1

                if rel_svm is not None:
                    rel_class = rel_svm.predict([vector])[0]

                    if rel_class == 1:
                        relevance = 0
                    else:
                        relevance = 1
            else:
                relevance = 2

                if irr_svm is not None:
                    irr_class = irr_svm.predict([vector])[0]

                    if irr_class == 1:
                        relevance = 2
                    else:
                        relevance = 3

        if relevance == 0:
            str_relevance = 'R+'
        elif relevance == 1:
            str_relevance = 'R'
        elif relevance == 2:
            str_relevance = 'I'
        else:
            str_relevance = 'I-'

        if img_id not in feedbacks:
            str_relevance = "Predicted as " + str_relevance
        else:
            if str_relevance != feedbacks[img_id][1]:
                str_relevance = "Overridden as " + str_relevance

        new_candidates.append((img_id, relevance, dist, str_relevance))

    return sorted(new_candidates, key=lambda x: (x[1], x[2]))