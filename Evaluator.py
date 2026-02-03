class Evaluator:

    @staticmethod
    def best_only(results, targets):
        accuracy = 0
        predictions = results.argmax(1)
        for i in range(targets.size(0)):
            if targets[i] == predictions[i]:
                accuracy += 1
        return accuracy

    @staticmethod
    def best_two(results, targets):
        accuracy = 0
        predictions_order = results.argsort(dim=1, descending=True)
        for i in range(targets.size(0)):
            if targets[i] == predictions_order[i][0] or targets[i] == predictions_order[i][1]:
                accuracy += 1
        return accuracy