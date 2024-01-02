from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import sklearn.metrics as sk_mt

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class MISTDataClass:
    def __init__(self, additional_data=None, model_type='random_forest'):  # конструктор
        self.DATA = fetch_openml('mnist_784', parser='pandas', as_frame=True).frame
        if additional_data is not None:
            self.add_data(additional_data)
        self.set_model(model_type)

    def add_data(self, data):
        data.columns = self.DATA.columns
        self.DATA = pd.concat((self.DATA, data), ignore_index=True)

    def set_model(self, model_type='random_forest'):
        self.model_name = model_type
        match model_type:
            case 'random_forest':
                self.model = RandomForestClassifier()
            case 'sgdc_classifier':
                self.model = SGDClassifier()
            case _:
                raise ValueError('invalid value:', model_type)

    # методы для визуализации
    def show(self):
        plt.figure()
        fig, axes = plt.subplots(nrows=3, ncols=6, figsize=(7, 4))
        fig.set(facecolor='0.75')
        for ax, loc in zip(*axes.reshape(1, -1), self.DATA.iloc):
            self.show_image(ax, self.get_image(loc), f'Training: {int(loc[-1])}')
        fig.suptitle(f'Size = {int(len(loc[:-1].tolist()) ** 0.5)} X'
                     f' {int(len(loc[:-1].tolist()) ** 0.5)}')
        plt.show()

    def show_image(self, ax, image, label=''):
        ax.set_axis_off()
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        ax.set_title(label)

    def info(self):
        print(self.DATA.info(), end='\n\n')
        print(self.DATA, end='\n\n')
        for i in range(10):
            count = len(self.DATA[self.DATA['class'] == i])
            print(f'Count {i}:  {count}')
        print()

    @staticmethod
    def get_image(loc):
        return np.array(loc[:-1].tolist()).reshape(28, 28)

    @staticmethod
    def get_image_x(X):
        return np.array(X.tolist()).reshape(28, 28)

    @staticmethod
    def get_x_y_loc(loc):
        return loc[:-1], loc[-1]

    @staticmethod
    def get_x_y_data(data):
        col_x, col_y = MISTDataClass.get_x_y_loc(data.columns)
        return data[col_x], data[col_y]

    # подгатовка данных для модели
    def prepare_data(self, shaffle=True, to_1 = True, as_num = False):
        '''подготавливаем данные для модели'''
        self.setting_data(to_1, as_num)
        self.split_train_test(shaffle)

    def split_train_test(self, shaffle: bool):
        self.test_set, self.train_set = train_test_split(self.DATA, test_size=6/7,
                                                         random_state=13, shuffle=shaffle)

    def setting_data(self, to_1: bool, as_num: bool):
        if to_1:
            self.DATA[self.DATA.columns[:-1]] = self.DATA[self.DATA.columns[:-1]] / 255
        if as_num:
            self.DATA['class'] = pd.to_numeric(self.DATA['class'])

    # работа с моделью
    def predict(self, test_X=None):
        if test_X is None:
            test_X, _ = self.get_x_y_data(self.test_set)
        return self.model.predict_proba(test_X)

    def training_model(self):
        train_X, train_Y = self.get_x_y_data(self.train_set)
        print('start training model')
        self.model.fit(train_X, train_Y)
        print('finish training model')

    def test_model(self):
        #TODO revork
        test_X, test_Y = self.get_x_y_data(self.test_set)
        pred = self.predict(test_X)
        loss = self.loss(test_Y, pred)
        print(self.model_name, end='\n\t')
        print('Loss: ', loss[0], loss[1])

    def loss(self, y_true, y_pred):
        return 'log_loss', sk_mt.log_loss(y_true, y_pred)

    def show_test_head_results(self):
        plt.figure()
        fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(13, 10))
        fig.set(facecolor='0.75')
        test_X, test_Y = self.get_x_y_data(self.test_set)
        for ax, X, Y in zip(*axes.reshape(1, -1), test_X.iloc, test_Y.iloc):
            st = ''
            for i in self.predict([X])[0]:
                st += str(i) + ', '
            self.show_image(ax, self.get_image_x(X), st + f'\n{Y}')
        fig.suptitle('0 1 2 3 4 5 6 7 8 9')
        plt.show()








