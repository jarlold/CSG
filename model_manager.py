import datetime
import pickle
import keras as K
from pandas import read_csv
import feature_extractions as fe
from sklearn.preprocessing import MinMaxScaler
from hashlib import sha1


def load_model(model_name):
    with open("models/"+model_name, 'rb') as opn:
        return pickle.load(opn)


class BaseStockModel:
    """
    A base model to create other model classes with. get_layer and get_data should
    be overwritten in the child class with the appropriate functions.
    """
    def __init__(self):
        self.model_type = str(self.__class__).replace("<class \'model_manager.", '').replace("\'>", ' ')
        self.model_name = self.model_type+str(datetime.datetime.now())
        self.model = None
        self.x_train, self.y_train, self.x_test, self.y_test = None, None, None, None
        self.descale = None
        self.history = None

    def get_layers(self):
        raise TypeError("This is a blank BaseModel model, it has no layers")

    def get_data(self):
        raise TypeError("This is a blank BaseModel model, it has no training data")

    def write_train_log(self, batch_size, epochs):
        with open("models/train_log.txt", 'a') as opn:
            opn.write(
                """
Model Name: {}
Test Date: {}
SHA1s: {} : {}
Vloss: {}
Loss: {}
Batch Size: {}
Epochs: {}\n\n
                """.format(
                    self.model_name,
                    datetime.datetime.now(),
                    sha1(self.x_train).hexdigest(), sha1(self.y_train).hexdigest(),
                    self.history['val_loss'][-1], self.history["loss"][-1],
                    batch_size, epochs
                ))

    def train(self, batch_size, epochs):
        self.get_data()
        self.model = self.get_layers()
        self.history = self.model.fit(
            self.x_train, self.y_train,
            validation_data=(self.x_test, self.y_test),
            batch_size=batch_size,
            epochs=epochs,
            shuffle=False,
        ).history
        self.write_train_log(batch_size, epochs)

    def save(self):
        with open("models/"+self.model_name, 'wb') as opn:
            pickle.dump(self, opn)


class PureHistoryLSTM(BaseStockModel):
    """
    A single layer LSTM network that works only off of stock history
    """
    def __init__(self, stockname, stock_ticker, timeframe_width):
        BaseStockModel.__init__(self)
        self.stock_name, self.stock_ticker = stockname, stock_ticker
        self.timeframe_width = timeframe_width

    def get_layers(self):
        model = K.models.Sequential()
        model.add(K.layers.LSTM(128, input_shape=(self.timeframe_width, 4)))
        model.add(K.layers.Activation("relu"))
        model.add(K.layers.Dense(4))
        model.add(K.layers.Activation('tanh'))
        opt = K.optimizers.Adam(learning_rate=0.001) # maybe try learning_rate=0.01
        model.compile(optimizer=opt, loss=K.losses.mean_squared_error, metrics=['mean_squared_error'])
        return model

    def get_data(self):
        data = read_csv("StockCSVs/"+self.stock_ticker+".csv")
        data = data.filter(items=[
            "Open", "High", "Low", "Close"
        ])
        data = data.to_numpy()
        scale = MinMaxScaler()
        scale.fit(data)
        scaled_data = scale.transform(data)
        self.descale = scale.inverse_transform
        x, y = fe.make_time_slices(scaled_data, scaled_data)

        self.x_train = x[:int(len(x)*0.75)]
        self.y_train = y[:int(len(y)*0.75)]

        self.y_test = y[int(len(y)*0.75):]
        self.x_test = x[int(len(x)*0.75):]


class PHLSTMTradeByMonth(PureHistoryLSTM):
    def get_data(self):
        data = read_csv("StockCSVs/"+self.stock_ticker+".csv")
        data = data.filter(items=[
            "Open", "High", "Low", "Close"
        ])
        data = data.to_numpy()
        scale = MinMaxScaler()
        scale.fit(data)
        scaled_data = scale.transform(data)
        self.descale = scale.inverse_transform
        x, y = fe.make_time_slices(scaled_data, scaled_data)

        self.x_train = x[:int(len(x)*0.75)]
        self.y_train = y[:int(len(y)*0.75)]

        self.y_test = y[int(len(y)*0.75):]
        self.x_test = x[int(len(x)*0.75):]



