#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest
import sklearn.metrics
from predictive_model import PredictiveModel
from sklearn.ensemble import RandomForestRegressor
import os

import warnings


def create_testing_file():
    testset = """
        {"timestamp": 1459926000, "ftr_vector": [1, 2, 3]}
        {"timestamp": 1459929600, "ftr_vector": [2, 3, 4]}
        {"timestamp": 1459933200, "ftr_vector": [1, 2, 3]}
        {"timestamp": 1459936800, "ftr_vector": [2, 3, 4]}
        {"timestamp": 1459940400, "ftr_vector": [1, 2, 3]}
        {"timestamp": 1459944000, "ftr_vector": [2, 2, 3]}
        {"timestamp": 1459947600, "ftr_vector": [1, 3, 2]}
        """

    #subdir = '../nrgStream-fusion/data'
    #subdir = './test/data'
    subdir= os.path.join('.', 'test', 'data')
    if not os.path.isdir(subdir):
        os.makedirs(subdir)
        
    filename = "N1_1h.json"
    filepath = os.path.join(subdir, filename)

    with open(filepath, 'w') as data_file:
        data_file.write(testset)

    return filepath

def create_model_instance():
        algorithm = "sklearn.ensemble.RandomForestRegressor(n_estimators=100, n_jobs=16, random_state=0)"
        sensor = "N1"
        horizon = 1
        evaluation_period = 72
        evaluation_split_point = 0.8
        error_metrics = [
            {'name': "R2 Score", 'short': "r2", 'function': sklearn.metrics.r2_score},
            {'name': "Mean Absolute Error", 'short': "mae", 'function': sklearn.metrics.mean_absolute_error},
            {'name': "Mean Squared Error", 'short': "mse", 'function': sklearn.metrics.mean_squared_error},
            {'name': "Root Mean Squared Error", 'short': "rmse", 'function': None}
        ]
        model = PredictiveModel(algorithm, sensor, horizon, evaluation_period, error_metrics, evaluation_split_point)

        return model


class SimpleWidgetTestCase(unittest.TestCase):

    def setUp(self):
        self.model = create_model_instance()


class TestClassProperties(SimpleWidgetTestCase):
    
    def test_sensor(self):
        self.assertEqual(self.model.sensor, "N1")

    def test_horizon(self):
        self.assertEqual(self.model.horizon, 1)

    def test_eval_periode(self):
        self.assertEqual(self.model.eval_periode, 72)

    def test_split_point(self):
    	self.assertEqual(self.model.split_point, 0.8)


class TestModelFunctionality(SimpleWidgetTestCase):

    def test_fit(self):

        # fit the model
        f = create_testing_file()
        score = self.model.fit(f)

        # check evaluation results
        self.assertAlmostEqual(score['mse'], 0.21, 2)
        self.assertAlmostEqual(score['rmse'], 0.46, 2)
        self.assertAlmostEqual(score['mae'], 0.38, 2)
        self.assertAlmostEqual(score['r2'], 0.15, 2)

        # clean up
        os.remove(f)

    def test_predict(self):

        # fit the model
        f = create_testing_file()
        self.model.fit(f)

        # make prediction
        prediction = self.model.predict([[1, 1, 1]])

        # check if prediction is valid
        self.assertEqual(prediction[0], 1.96)        
        os.remove(f)


class TestModelSerialization(SimpleWidgetTestCase):

    def test_save(self):

        # file names
        model_file = os.path.join('.', 'test', 'data', 'model')
        dataset_file = create_testing_file()
        
        # first test if we get exception when trying to use unfitted model
        with self.assertRaises(Exception) as context:
            self.model.predict([[1, 1, 1]])
            self.assertTrue("not fitted yet" in str(context.exception))

        # fit the model
        self.model.fit(dataset_file)

        # test if prediction works
        prediction = self.model.predict([[1, 1, 1]])
        self.assertEqual(prediction[0], 1.96)

        # save the model
        self.model.save(model_file)

        # check if file was created
        self.assertTrue(os.path.exists(model_file))

        # clean up
        os.remove(model_file)
        os.remove(dataset_file)

    def test_load(self):

        # file names
        model_file = os.path.join('.', 'test', 'data', 'model')
        dataset_file = create_testing_file()
        
        # create saved file of dummy model
        dummy_model = create_model_instance()
        dummy_model.fit(dataset_file)
        dummy_model.save(model_file)

        # first test if we get exception when trying to use unfitted model
        with self.assertRaises(Exception) as context:
            self.model.predict([[1, 1, 1]])
            self.assertTrue("not fitted yet" in str(context.exception))

        # load the model
        self.model.load(model_file)

        # test if prediction works
        prediction = self.model.predict([[1, 1, 1]])
        self.assertEqual(prediction[0], 1.96)

        # clean up
        os.remove(model_file)
        os.remove(dataset_file)


class TestModelEvaluation(SimpleWidgetTestCase):

    def test_evaluation_warning(self):

        # test if "not enough predictions" warning will be thrown
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            score = self.model.evaluate({'value': 1}, 1)

            self.assertEqual(len(w), 1)
            self.assertEqual(w[-1].category, UserWarning)
            self.assertTrue("Not enough predictions" in str(w[-1].message))

    def test_evaluation_score(self):

        # send model enough predictions to fill the buffers
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            for a in range(1,80):
                output = self.model.evaluate({'value': 0}, 1)

        # check evaluation results
        self.assertEqual(output['mse'], 1)
        self.assertEqual(output['rmse'], 1)
        self.assertEqual(output['mae'], 1)
        self.assertEqual(output['r2'], 0)
        
    def test_perfect_score(self):

        # send model enough predictions to fill the buffers
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            for a in range(1,80):
                output = self.model.evaluate({'value': 1}, 1)

        # check evaluation results
        self.assertEqual(output['mse'], 0)
        self.assertEqual(output['rmse'], 0)
        self.assertEqual(output['mae'], 0)
        self.assertEqual(output['r2'], 1)

    def test_evaluation_buffers(self):

        # send model enough predictions to fill the buffers
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            for a in range(1,100):
                self.model.evaluate({'value': 1}, 1)

        # check buffers
        self.assertEqual(len(self.model.measurements), self.model.eval_periode)
        self.assertEqual(len(self.model.predictions), self.model.eval_periode + self.model.horizon)
    
    def test_predictability_index(self):

        # fit the model
        f = create_testing_file()
        score = self.model.fit(f)

        # test predictability index
        self.assertAlmostEqual(self.model.predictability, score['r2']*100, 0)

        # clean up
        os.remove(f)
    
if __name__ == '__main__':
    unittest.main(verbosity=2)
