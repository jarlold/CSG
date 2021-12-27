import numpy as np
from hashlib import sha1
import datetime


class BaseTest:
    """
    A base class for creating test-methods. test_function should be replaced with a testing
    mechanism (which should fill self.results as well as return the results). results_format_string
    should be replaced with a formatting template that will be called with the result(s).
    (I.E "Results were: {}")
    """
    def __init__(self, csg_model, x=None, y=None):
        self.csg_model = csg_model
        self.model = csg_model.model
        self.x = csg_model.x_test if x is None else x
        self.y = csg_model.y_test if y is None else y
        self.results = None
        self.result_format_string = None

    def test_function(self):
        raise TypeError("This is a blank BaseTest, it has no testing function.")

    def write_res(self):
        if self.results.__class__ == list:
            s = self.result_format_string.format(*self.results)
        else:
            s = self.result_format_string.format(self.results)
        with open("./models/test_log.txt", 'a') as opn:
            opn.write(
                """
Model Name: {}
Test Date: {}
SHA1s: {} : {}
Test Results:
{}\n\n
                """.format(
                    self.csg_model.model_name,
                    datetime.datetime.now(),
                    sha1(self.x).hexdigest(), sha1(self.y).hexdigest(),
                    s
                )
            )

    def print_res(self):
        if self.results.__class__ == list:
            s = self.result_format_string.format(*self.results)
        else:
            s = self.result_format_string.format(self.results)
        print(s)
        self.write_res()


class OpenCloseDeltaTrade(BaseTest):
    def __init__(self, csg_model, x=None, y=None, open_pos=0, close_pos=3):
        BaseTest.__init__(self, csg_model, x=x, y=y)
        self.open_pos, self.close_pos = open_pos, close_pos
        self.accumulated_delta = 0
        self.possible_accumulated_delta = 0
        self.result_format_string =\
            "Accumulated Opn-Cls Delta: {}\nPossible Accumulated Opn-Cls Delta: {}"

    def test_function(self):
        # Shape the time slices to each be [-1, width, channels]
        s1, _, _ = self.x.shape
        reshaped_x = self.x.reshape([s1, 1, *self.x.shape[1:]])
        # STart iterating through the samples
        for time_slice, base in zip(reshaped_x, self.y):
            # Make a prediction and de-scale it
            prediction = self.csg_model.model.predict(time_slice)
            prediction = self.csg_model.descale(prediction).flatten()
            # If the open-close difference is positive, add it to the possible_accumulated_delta
            b = base.reshape([-1, base.shape[0]])
            b = self.csg_model.descale(b).flatten()
            actual_opn_cls_delta = b[self.close_pos] - b[self.open_pos]
            if actual_opn_cls_delta > 0:
                self.possible_accumulated_delta += actual_opn_cls_delta
            # If the prediction says the closing price will be higher than the opening
            # then add the de-scaled difference to the accumulated difference
            if prediction[self.open_pos] < prediction[self.close_pos]:
                self.accumulated_delta += actual_opn_cls_delta

        self.results = [self.accumulated_delta, self.possible_accumulated_delta]
        return [self.accumulated_delta, self.possible_accumulated_delta]


class PureDifference(BaseTest):
    """
    Find the average absolute delta between the predicted result and the base-values.
    """
    def __init__(self, model, x, y):
        BaseTest.__init__(self, model, x, y)
        self.result_format_string = "Pure Difference [f=abs(x1-x2)]: {}"

    def test_function(self):
        deltas = []
        for i, j in zip(self.x, self.y):
            i = i.reshape(-1, *i.shape)
            p = self.model.predict(i).flatten()
            deltas.append(abs(p - j))
        s = np.array(deltas).sum()
        res = float(s) / float(len(deltas))
        self.results = res
        return res


class Bias(BaseTest):
    """
    Find the average delta between the predicted result and the base-values.
    Can let you know if the network typically over-shoots or undershoots.
    """
    def __init__(self, model, x, y):
        BaseTest.__init__(self, model, x, y)
        self.result_format_string = "Bias [f=x1-x2]: {}"

    def test_function(self):
        deltas = []
        for i, j in zip(self.x, self.y):
            i = i.reshape(-1, *i.shape)
            p = self.model.predict(i).flatten()
            deltas.append(p - j)
        s = np.array(deltas).sum()
        res = float(s) / float(len(deltas))
        self.results = res
        return res

