import pandas as pd
from numpy.random import RandomState
import numpy as np
from tabulate import tabulate
from random import randint
import matplotlib.pylab as plt
import scipy.stats as stats

class WeatherPrediction():
    def __init__(self, file, attribute):
        self.data = pd.read_csv(file, header=(0), delimiter=";")
        self.data = self.data.drop(columns=["dt","dt_iso","timezone","city_name", "wind_speed","lat","lon","sea_level","grnd_level","weather_icon",'wind_deg','rain_3h', 'snow_1h', 'snow_3h', "clouds_all","weather_description","weather_id", "temp_min", "temp_max","feels_like"])
        self.data = self.convert_to_numerical(self.data)
        self.data = self.add_play_column(self.data)
        self.attribute = attribute
        self.priori = {}
        self.training, self.test = self._create_sets(self.data, train_size=0.7)
        self.X_train = self.training.iloc[:, :-1]
        self.X_test = self.test.iloc[:, :-1]
        self.y_train = self.training.iloc[:,-1]
        self.y_test = self.test.iloc[:,-1]
        self.random_predict = []
    
    def convert_to_numerical(self, data):
        for attribute in ["temp","pressure", "humidity","rain_1h"]:
            if attribute == "temp":
                data['temp'] = data["temp"].apply(lambda x: float("".join(str(x)[:3])))
                plt.xlabel("temperature [K]")
            elif attribute =="wind_speed":
                data[attribute] = data[attribute].apply(lambda x: int(x))
            else:
                data[attribute] = data[attribute].apply(lambda x: float(x))
            
            if attribute == "pressure":
                plt.xlabel("pressure [hPa]")
            elif attribute == "humidity":
                plt.xlabel("humidity [%]")
            elif attribute == "rain_1h":
                plt.xlabel("rain [%]")   
            #data[attribute].plot.hist(bins=30, alpha=0.5, color="grey")
            #plt.show()
            #plt.clf()
        return data
    
    def add_play_column(self, data):
        data.loc[(data.weather_main == "Drizzle") | (data.weather_main == "Fog") \
        | (data.weather_main == "Haze")| (data.weather_main == "Mist")| (data.weather_main == "Rain")| (data.weather_main == "Thunderstorm") \
        | (data.weather_main == "Smoke")| (data.weather_main == "Dust")| (data.weather_main == "Squall") , 'play'] = float(0)
        data.loc[(data.weather_main == "Clear")| (data.weather_main == "Clouds")| (data.weather_main == "Snow"), 'play'] = float(1)
        data["rain_1h"] = data["rain_1h"].fillna(float(0))
        data.to_csv("MEINE_DATEN.csv")
        data = data.drop(columns=["weather_main"])
        data = data.dropna()
        return data
    
    def _create_sets(self, data, train_size=0.7):
        """Randomly split data in training (70%) and test set.

        Parameters
        ----------
        data : DataFrame
            Weather data.

        Returns
        -------
        DataFrames
            Subsets (training, test) of weather data.
        """
        train = data.sample(frac=train_size, random_state=RandomState())
        test = data.loc[~data.index.isin(train.index)]
        return train, test
    
    def calculate_mean_and_var(self, X, y):
        self.mean = X.groupby(y).apply(np.mean).to_numpy()
        self.var = X.groupby(y).apply(np.var).to_numpy()


    def calculate_priori(self, X, y):
        self.priori = (X.groupby(y).apply(lambda x: len(x))/ self.num_of_rows).to_numpy()

    def gaussian(self, i, x):
        mean = self.mean[i]
        var = self.var[i]
        numerator = np.exp((-1/2)*((x-mean)**2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        prob = numerator / denominator
        return prob

    def calculate_posterior(self, X):
        posteriors = []
        for i in range(self.num_of_classes):
            priori = np.log(self.priori[i])
            conditional = np.sum(np.log(self.gaussian(i, X)))
            posterior = priori + conditional
            posteriors.append(posterior)
            print(i,X,priori, conditional, posterior, posteriors)
        result = self.classes[np.argmax(posteriors)]
        print("Result",result)
        return result
    
    def plot_likelihood(self):
        for i, m in enumerate(self.mean):
            for j, v in enumerate(self.mean[i]):
                #print(self.mean, self.var)
                x_lower_limit = min([ self.mean[0][j] - 3*(self.var[0][j])**(1/2) ,  self.mean[0][j] - 3*(self.var[0][j])**(1/2) ])
                x_upper_limit = max([ self.mean[0][j] + 3*(self.var[0][j])**(1/2) ,  self.mean[0][j] + 3*(self.var[0][j])**(1/2) ])
                x_values = np.linspace(x_lower_limit,x_upper_limit,1000)
                plt.plot(x_values, stats.norm.pdf(x_values, self.mean[0][j], (self.var[0][j])**(1/2)),label="not play "+r"$\mu = $"+ f"{round(self.mean[0][j],2)}, "+r"$\sigma = $"+f"{round(self.var[0][j],2)}")
                plt.plot(x_values, stats.norm.pdf(x_values, self.mean[1][j], (self.var[1][j])**(1/2)), label="play "+r"$\mu = $"+ f"{round(self.mean[1][j],2)}, "+r"$\sigma = $"+f"{round(self.var[1][j],2)}")
                plt.legend()
                if j == 0:
                    plt.savefig("temp.png")
                elif j ==1 :
                    plt.savefig("pressure.png")
                elif j ==2 :
                    plt.savefig("humidity.png")
                else:
                    plt.savefig("rain.png")
                plt.show()
                plt.clf()



    def fit(self):
        # print(self.X_train,self.y_train)
        # print(self.X_test, self.y_test)
        features = self.X_train
        target = self.y_train
        self.classes = np.unique(target)# [0.0, 1.0]
        self.num_of_classes = len(self.classes)
        self.num_of_rows = features.shape[0]
        self.num_of_features = features.shape[1]

        self.calculate_mean_and_var(features, target)
        self.calculate_priori(features, target)
    
    def predict(self):
        x = self.X_test
        predictions = [self.calculate_posterior(att) for att in x.to_numpy()]
        self.predictions = predictions
        for x in range(0,len(predictions)):
            self.random_predict.append(randint(0,1))
        self.random_predict = [float(p) for p in self.random_predict]
        self.plot_likelihood()
    
    def random_set_accuracy(self):
        self.y_test = list(self.y_test)
        correct_prediction = 0
        print(len(self.random_predict))
        for i, pred in enumerate(list(self.random_predict)):
            if self.y_test[i] == pred:
                correct_prediction +=1
        self.random_acc = correct_prediction/len(self.y_test)

    def accuracy(self):
        correct_zero = 0
        correct_one = 0
        ones = 1
        zeros = 1
        self.y_test = list(self.y_test)
        for i, pred in enumerate(list(self.predictions)):
            if self.y_test[i] == pred:
                if pred == 1.0:
                    correct_one += 1
                else:
                    correct_zero += 1
                
            if self.y_test[i] == 1.0:
                ones += 1
            else:
                zeros += 1
        acc = (correct_one+correct_zero)/len(self.y_test)

        info = {
            'Parameter': [
                    "Size of train set",
                    "Size of test set",
                    'A priori of "not play" in train set',
                    'A priori of "play" in train set',
                    'Mean of "not play" - temperature',
                    'Mean of "not play" - pressure',
                    'Mean of "not play" - humidity',
                    'Mean of "not play" - rain in past hour',
                    'Mean of "play" - temperature',
                    'Mean of "play" - pressure',
                    'Mean of "play" - humidity',
                    'Mean of "play" - rain in past hour',
                    'Predicted "not play" correct',
                    'Predicted "play" correct',
                    "Accuracy",
                    "Random decision accuracy"
                    ], 
        
        
            'Value': [  f"{len(self.X_train)} ({round(len(self.X_train)/(len(self.X_train)+len(self.X_test))*100,2)}%)", 
                    f"{len(self.X_test)} ({round(len(self.X_test)/(len(self.X_train)+len(self.X_test))*100,2)}%)",
                    f"{round((self.priori[0])*100,2)}%",
                    f"{round((self.priori[1])*100,2)}%",
                    round(self.mean[0][0],2),
                    round(self.mean[0][1],2),
                    round(self.mean[0][2],2),
                    round(self.mean[0][3],2),
                    round(self.mean[1][0],2),
                    round(self.mean[1][1],2),
                    round(self.mean[1][2],2),
                    round(self.mean[1][3],2),
                    f"{round((correct_zero/zeros)*100,2)}%",
                    f"{round((correct_one/ones)*100,2)}%",
                    f"{round(acc*100,2)}%",
                    f"{round(self.random_acc*100,2)}%"
                    ],
            "Variance":["","","","",
                    round(self.var[0][0],2),
                    round(self.var[0][1],2),
                    round(self.var[0][2],2),
                    round(self.var[0][3],2),
                    round(self.var[1][0],2),
                    round(self.var[1][1],2),
                    round(self.var[1][2],2),
                    round(self.var[1][3],2)]}
        print(tabulate(info, headers='keys',tablefmt="fancy_grid"))
        self.acc = acc
    
if __name__ == "__main__":
    WeatherClass = WeatherPrediction("weather.csv","play")
    WeatherClass.fit()
    WeatherClass.predict()
    WeatherClass.random_set_accuracy()
    WeatherClass.accuracy()

# 0 [ 276. 1004.   69.  113.    0.] -1.55 -21.35 -22.90
# 1 [ 276. 1004.   69.  113.    0.] -0.23 -17.55 -17.79 --> Result: 1.0



