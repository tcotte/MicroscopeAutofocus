class ClassificationMetrics:
    def __init__(self):
        self.true_positives = 0
        self.false_positives = 0
        self.false_negatives = 0

        self.reset()

    def update(self, preds, targets):
        self.true_positives += ((preds == 1) & (targets == 1)).sum().item()
        self.false_positives += ((preds == 1) & (targets == 0)).sum().item()
        self.false_negatives += ((preds == 0) & (targets == 1)).sum().item()

    def precision(self):
        """
        precision = TP / (TP + FP)
        :return:
        """
        return self.true_positives / (self.true_positives + self.false_positives) if (
               self.true_positives + self.false_positives) > 0 else 0

    def recall(self):
        """
        recall = TP / (TP + FNQ)
        :return:
        """
        return self.true_positives / (self.true_positives + self.false_negatives) if (
               self.true_positives + self.false_negatives) > 0 else 0

    def f1(self):
        recall = self.recall()
        precision = self.precision()

        return 2 * (recall * precision) / (recall + precision) if (precision + recall) > 0 else 0

    def reset(self):
        self.true_positives = 0
        self.false_positives = 0
        self.false_negatives = 0

    def compute(self):
        return {'precision': f'{self.precision():.2f}',
                'recall': f'{self.recall():.2f}',
                'f1': f'{self.f1():.2f}'}

    def __call__(self, preds, targets):
        self.update(preds, targets)
        return self.compute()
