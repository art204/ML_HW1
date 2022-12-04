from fastapi import FastAPI, Request, File, UploadFile
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import Response
from pydantic import BaseModel
from typing import List

import pandas as pd
import numpy as np
import re
from sklearn import impute
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge


class Predictor:

    def __init__(self):
        df_train = pd.read_csv(
            'https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_train.csv')
        df_train.drop_duplicates(subset=df_train.columns.drop('selling_price'), keep='first', inplace=True)
        df_train.reset_index(drop=True, inplace=True)

        pd.set_option('display.max_columns', None)
        X_train = df_train
        y_train = df_train['selling_price']

        self.__real_mis_replacer = None
        self.__obj_mis_replacer = None
        self.__raw_columns = df_train.columns
        self.__train_columns = None
        self.__scaler = None
        self.__scaled_columns = ['year', 'km_driven', 'mileage', 'engine', 'max_power', 'max_torque_rpm', 'torque_nm']

        X_train = self.prep_data(X_train)

        # В конструкторе класса Predictor обучим модель, которую будем использовать для предсказания цены
        self.lr = Ridge(alpha=8, random_state=42)
        self.lr.fit(X_train, y_train)

    # Подготовка данных: создание DataFrame из исходных данных,
    # приведение колонок к нужному типу, заполнение пропусков,
    # кодирование категориальных признаков one-hot encoding, нормализация признаков
    def prep_data(self, data):
        df = self.__get_dataframe(data)
        for col in ['mileage', 'engine', 'max_power']:
            df[col] = df[col].apply(self.get_float)

        df['max_torque_rpm'] = df['torque'].apply(self.get_max_torque_rpm)
        df['torque_nm'] = df['torque'].apply(self.get_torque)
        df.drop(columns=['name', 'torque', 'selling_price'], inplace=True)

        df['engine'] = df['engine'].astype('float64')
        df['seats'] = df['seats'].astype('float64')
        df['mileage'] = df['mileage'].astype('float64')
        df['max_power'] = df['max_power'].astype('float64')
        df['torque_nm'] = df['torque_nm'].astype('float64')
        df['max_torque_rpm'] = df['max_torque_rpm'].astype('float64')

        cat_features_mask = (df.dtypes == "object").values
        df_real = df[df.columns[~cat_features_mask]]
        if self.__real_mis_replacer is None:
            self.__real_mis_replacer = impute.SimpleImputer(strategy='median')
            self.__real_mis_replacer.fit(df_real)
        df_real = pd.DataFrame(data=self.__real_mis_replacer.transform(df_real), columns=df_real.columns)

        df_cat = df[df.columns[cat_features_mask]]
        if self.__obj_mis_replacer is None:
            self.__obj_mis_replacer = impute.SimpleImputer(strategy='most_frequent')
            self.__obj_mis_replacer.fit(df_cat)
        df_cat = pd.DataFrame(data=self.__obj_mis_replacer.transform(df_cat), columns=df_cat.columns)

        df = pd.concat([df_real, df_cat], axis=1)

        df['engine'] = df['engine'].astype('int64')
        df['seats'] = df['seats'].astype('int64')

        if self.__train_columns is not None:
            df = pd.get_dummies(df, drop_first=False,
                                columns=['fuel', 'seller_type', 'transmission', 'owner', 'seats'])
            pd.set_option('display.max_columns', None)

            for col in df.columns:
                if col not in self.__train_columns:
                    df.drop(columns=[col], inplace=True)

            for i in range(len(self.__train_columns)):
                col = self.__train_columns[i]
                if col not in df:
                    df.insert(i, col, 0)

        else:
            df = pd.get_dummies(df, drop_first=True,
                                columns=['fuel', 'seller_type', 'transmission', 'owner', 'seats'])
            self.__train_columns = df.columns

        if self.__scaler is None:
            self.__scaler = StandardScaler()
            self.__scaler.fit(df[self.__scaled_columns])

        df[self.__scaled_columns] = self.__scaler.transform(df[self.__scaled_columns])

        return df

    @staticmethod
    def get_float(x):
        if pd.isna(x) or x == 'nan':
            return None

        arr = re.findall(r'\d+', x)
        if len(arr) == 0:
            return None
        else:
            return float('.'.join(arr))

    @staticmethod
    def get_max_torque_rpm(string):
        if pd.isna(string) or string == 'nan':
            return None

        arr = re.findall(r'\s.+', string)
        if len(arr) > 0:
            if '~' in arr[0]:
                digits = re.findall(r'\d+', arr[0].strip().split('~')[-1])
            else:
                digits = re.findall(r'\d+', arr[0].strip().split('-')[-1])
            return float(''.join(digits))

    @staticmethod
    def get_torque(string):
        if pd.isna(string) or string == 'nan':
            return None
        string = string.lower()
        arr = re.findall(r'\d+\.?\d*[\s@nk(]', string)
        torque = '.'.join(re.findall(r'\d+', arr[0]))
        if 'nm' in string:
            return float(torque)
        elif 'kgm' in string:
            return float(torque) * 9.80665
        else:
            return float(torque)

    # Если входные данные не DataFrame, то создает DataFrame (только для Item или List[Item])
    def __get_dataframe(self, obj):
        if isinstance(obj, pd.DataFrame):
            return obj.copy()
        if isinstance(obj, Item):
            df = pd.DataFrame()
            d = obj.dict()
            for col in self.__raw_columns:
                df.loc[0, col] = d[col]
            return df
        if isinstance(obj, List):
            df = pd.DataFrame()
            for i in range(len(obj)):
                d = obj[i].dict()
                for col in self.__raw_columns:
                    df.loc[i, col] = d[col]
            return df

    # Предсказание стоимости машины
    def predict(self, data):
        df = self.prep_data(data)
        result = self.lr.predict(df)
        return result


app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

predictor = Predictor()


class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


class Items(BaseModel):
    objects: List[Item]

@app.get("/upload")
async def upload(request: Request):
    return templates.TemplateResponse('index.html', {"request": request})

# Предсказать стоимость объектов, переданных в csv файле.
# Результат вернуть в csv файле.
@app.post("/predict_csv")
async def predict_csv(request: Request,
                 data_file: UploadFile = File(...)):
    save_path = 'static/predict_data.csv'
    with open(save_path, 'wb') as f:
        for line in data_file.file:
            f.write(line)
    df = pd.read_csv(save_path)
    prediction = predictor.predict(df)
    df['price_prediction'] = np.round(prediction)
    return Response(df.to_csv(index=False), media_type='text/csv')

# Предсказать стоимость одного объекта
@app.post("/predict_item")
def predict_item(item: Item) -> float:
    return round(predictor.predict(item)[0])

# Предсказать стоимость списка объектов
@app.post("/predict_items")
def predict_items(items: List[Item]) -> List[float]:
    return [round(num) for num in predictor.predict(items)]
