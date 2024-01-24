class PerLabelPerf:
    """
    Multiclass (string label) performance calculator.
    It considers classifications as correct even if one
    label is a prefix of another labe.
    For example:
    pred: Faces, actual: Faces_easy
    this is considered as a correct prediction, and vice versa.
    https://www.evidentlyai.com/classification-metrics/multi-class-metrics
    """

    def __init__(self, all_labels):
        self.TP = {}
        self.FP = {}
        self.FN = {}
        self.total = 0
        self.correct = 0
        for label in all_labels:
            self.TP[label] = 0
            self.FP[label] = 0
            self.FN[label] = 0


    def process_perf(self, y, y_pred):
        self.total += 1
        if y.startswith(y_pred) or y_pred.startswith(y):
            self.correct += 1
            self.TP[y] += 1
        else:
            self.FP[y_pred] += 1
            self.FN[y] += 1


    def get_precision(self, label):
        if self.TP[label] == 0 and self.FP[label] == 0:
            return 0
        return self.TP[label] / (self.TP[label] + self.FP[label])


    def get_recall(self, label):
        if self.TP[label] == 0 and self.FN[label] == 0:
            return 0
        return self.TP[label] / (self.TP[label] + self.FN[label])


    def get_f1_score(self, label):
        if self.get_precision(label) == 0 and self.get_recall(label) == 0:
            return 0
        return (2 * self.get_precision(label) * self.get_recall(label)) / (self.get_precision(label) + self.get_recall(label))


    def get_overall_accuracy(self):
        return self.correct / self.total