from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import sklearn.metrics as sk_mt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class MISTDataClass:
    def __init__(self, model_type='random_forest'):  # конструктор
        self.DATA = fetch_openml('mnist_784', parser='pandas', as_frame=True).frame
        self.set_model(model_type)

    def set_model(self, model_type='random_forest'):
        match model_type:
            case 'random_forest':
                self.model = RandomForestClassifier()
            case 'sgdc_classifier':
                self.model = SGDClassifier()
            case _:
                raise ValueError('invalid value:', model_type)

    # методы для визуализации
    def show(self):
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
    def prepare_data(self, shaffle: bool, to_1: bool):
        '''подготавливаем данные для модели'''
        self.setting_data(to_1)
        self.split_train_test(shaffle)

    def split_train_test(self, shaffle: bool):
        self.test_set, self.train_set = train_test_split(self.DATA, test_size=60_000, train_size=10_000,
                                                         random_state=13, shuffle=shaffle)

    def setting_data(self, to_1: bool):
        if to_1:
            self.DATA[self.DATA.columns[:-1]] = self.DATA[self.DATA.columns[:-1]] / 255
        self.DATA['class'] = pd.to_numeric(self.DATA['class'])

    # работа с моделью
    def training_model(self):
        train_X, train_Y = self.get_x_y_data(self.train_set)
        print('start training model')
        self.model.fit(train_X, train_Y)
        print('finish training model')

    def test_model(self):
        #TODO revork
        test_X, test_Y = self.get_x_y_data(self.test_set)
        pred = self.model.predict(test_X)
        print((test_Y != pred).mean())
        for name, func in zip(['MSE', 'RMSE', 'MAE'],
                              [sk_mt.mean_squared_error, sk_mt.mean_squared_error,
                               sk_mt.mean_absolute_error]):
            e = func(test_Y, pred) if name != 'RMSE' else func(test_Y, pred)**0.5
            print(f'{name}: {e}')

    def error(self):
        pass
    def show_test_head_results(self):
        fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10, 10))
        fig.set(facecolor='0.75')
        test_X, test_Y = self.get_x_y_data(self.test_set)
        for ax, X, Y in zip(*axes.reshape(1, -1), test_X.iloc, test_Y.iloc):
            self.show_image(ax, self.get_image_x(X), str(*self.model.predict_proba([X])) + f'\n{Y}')
        plt.show()







