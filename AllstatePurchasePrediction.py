__author__ = 'Pawel Kozela'


class AllstatePurchasePredictor():

    def __init__(self):
        pass

    def train(self, data, headers, answers):
        pass

    def predict(self, data, headers):
        answers = []

        for i, key in enumerate(data):
            last_values = ''.join(data[key][-1][headers['A']:headers['G']+1])
            ans = ','.join([key, last_values])
            answers.append(ans)

        return answers