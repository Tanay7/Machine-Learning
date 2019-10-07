import pickle
import collections
import json

# Missing columns handled by taking Mean of their respective Training set (refer metadata.xlsx)
class RandomForestModel:

    def __init__(self, filename, features_to_predict):

        self.rf_model = pickle.load(open(filename, 'rb'))
        self.features_to_predict = features_to_predict
        self.no_records = 0
        self.result = {}
        self.mean_val = {'CRIM': 1.1554, 'ZN': 14.87704918, 'INDUS': 8.882431694, 'CHAS': 0.07650273224, 'NOX': 0.520,
                         'RM': 6.317, 'AGE': 62.007, 'DIS': 4.3946, 'RAD': 5.587, 'TAX': 331.357,
                         'PTRATIO': 17.963, 'BLACK': 379.659, 'LSTAT': 10.976}
        self.result['result'] = {'predictions': []}

    def input_record_validator(self, record):

        try:

            if type(record) is dict:

                self.no_records += 1

            else:

                raise ValueError

            return True

        except ValueError:

            self.result['status'] = "Failed"
            self.result['message'] = "Incorrect input format. Please make sure json file is valid."
            self.result['error'] = "Invalid file format. Error in record {}".format(self.no_records + 1)

            self.no_records = 0

            return False

    def handling_missing_values(self, each_record):

        missing_val = list(set(self.mean_val.keys()) - set(each_record.keys()))

        for val in missing_val:

            each_record[val] = self.mean_val[val]

        return each_record

    def pre_processing_data(self, each_record):

        ordered_record = collections.OrderedDict()

        each_record = self.handling_missing_values(each_record)

        for val in self.mean_val.keys():

            ordered_record[val] = each_record[val]

        return ordered_record

    def prediction(self, each_record):

        predicted_val = self.rf_model.predict([list(each_record.values())])

        return predicted_val

    def main_process(self):

        res = []

        for each_record in self.features_to_predict:

            if self.input_record_validator(each_record):

                try:

                    each_record = self.pre_processing_data(each_record)

                    res.append(self.prediction(each_record)[0])

                except Exception as e:

                    self.result['status'] = "Failed"
                    self.result['message'] = "Something went wrong!"
                    self.result['error'] = "Error: {}".format(e)

            else:

                return self.result

        self.result['status'] = "Success"
        self.result['message'] = "Done"
        self.result['error'] = "NA"
        self.result['result']['predictions'] = res

        return self.result
