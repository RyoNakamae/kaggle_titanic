# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# Python3

import pprint

import pandas as pd
from sklearn import tree
import numpy as np


csv_train = r'C:\PJ\20180917_kaggle\kaggle_titanic\files\train.csv'
csv_test = r'C:\PJ\20180917_kaggle\kaggle_titanic\files\test.csv'


def read_dataset(df):
   
    # サイズ
    print("= size =")
    pprint.pprint(df.shape)

    # 統計量
    print("= describe =")
    pprint.pprint(df.describe())

    # 欠損データ
    print("= 欠損データ =")
    pprint.pprint(kesson_table(df))

    # 先頭データ
    print("= head =")
    pprint.pprint(df.head(10))

def read_datasets(df_train, df_test):
    print("---> train ---")
    read_dataset(df_train)
    print("---> test ---")
    read_dataset(df_test)
    print("------------------------")

def kesson_table(df): 
    null_val = df.isnull().sum()
    percent = 100 * df.isnull().sum()/len(df)
    kesson_table = pd.concat([null_val, percent], axis=1)
    kesson_table_ren_columns = kesson_table.rename(
    columns = {0 : '欠損数', 1 : '%'})
    return kesson_table_ren_columns

def distribution(df, column_name):
    #平均・標準偏差・null数を取得する
    Age_average = df[column_name].mean() #平均値
    Age_std = df[column_name].std()  #標準偏差
    Age_nullcount = df[column_name].isnull().sum() #null値の数＝補完する数

    # 正規分布に従うとし、標準偏差の範囲内でランダムに数字を作る
    rand = np.random.randint(Age_average - Age_std, Age_average + Age_std , size = Age_nullcount)

    #Ageの欠損値
    df[column_name][np.isnan(df[column_name])] = rand
    pass

def pre_processing(df):
    # 欠損値の補完
    ## 中央値を設定：median()で中央値が取得できる
    df["Age"] = df["Age"].fillna(df["Age"].median())
    # distribution(df, "Age")
    ## 最頻値を設定:pandas.Seriesに対してmode()で最頻値のSeriesが取得できるのでそれの先頭を使用
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    # distribution(df, "Embarked")
    ## 最頻値を設定
    df["Fare"] = df["Fare"].fillna(df["Fare"].mode()[0])
    # distribution(df, "Fare")

    # 文字列→数字化
    df['Sex'] = df['Sex'].map({ 'male':0, 'female':1})
    df['Embarked'] = df['Embarked'].map({ 'S':0, 'C':1, 'Q':1})
    # df['Sex'] = df['Sex'].map({ 'male':1, 'female':0})
    # df['Embarked'] = df['Embarked'].map({ 'S':1, 'C':2, 'Q':0})

def output_file(prediction, df_test, filename):
    # PassengerIdを取得
    PassengerId = np.array(df_test["PassengerId"]).astype(int)
    # prediction(予測データ）とPassengerIdをデータフレームへ落とし込む
    df_solution_tree = pd.DataFrame(prediction, PassengerId, columns = ["Survived"])
    # csvファイルとして書き出し
    df_solution_tree.to_csv(filename, index_label = ["PassengerId"])
    pass

if __name__ == '__main__':

    print("start")

    # ファイル読み込み
    df_train = pd.read_csv(csv_train)
    df_test = pd.read_csv(csv_test)

    # データの確認
    # read_datasets(df_train, df_test)

    # pre_processing
    # pd.options.mode.chained_assignment = None
    pre_processing(df_train)
    pre_processing(df_test)
    # pd.options.mode.chained_assignment = 'warn'

    # 変換後データの確認
    # read_datasets(df_train, df_test)


    # 「train」の目的変数と説明変数の値を取得
    target = df_train["Survived"].values
    # features_one = df_train[["Pclass", "Sex", "Age", "Fare"]].values
    # 決定木の作成
    # my_tree_one = tree.DecisionTreeClassifier()
    # my_tree_one = my_tree_one.fit(features_one, target)
    # 「test」の説明変数の値を取得
    # test_features = df_test[["Pclass", "Sex", "Age", "Fare"]].values
    # 「test」の説明変数を使って「my_tree_one」のモデルで予測
    # my_prediction = my_tree_one.predict(test_features)

    # 予測結果をファイルに出力
    # output_file(my_prediction ,df_test , "my_tree_one.csv")



    # 追加となった項目も含めて予測モデルその2で使う値を取り出す
    # lst_params = ["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]
    lst_params = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    # features_two = df_train[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values
    features_two = df_train[lst_params].values
    # 決定木の作成とアーギュメントの設定
    max_depth = 10
    min_samples_split = 5
    my_tree_two = tree.DecisionTreeClassifier(max_depth = max_depth, min_samples_split = min_samples_split, random_state = 1)
    my_tree_two = my_tree_two.fit(features_two, target)

    # tsetから「その2」で使う項目の値を取り出す
    # test_features_2 = df_test[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values
    test_features_2 = df_test[lst_params].values
    # 「その2」の決定木を使って予測
    my_prediction_tree_two = my_tree_two.predict(test_features_2)

    # 予測結果をファイルに出力
    output_file(my_prediction_tree_two ,df_test , "my_tree_two.csv")

    print("end")
    pass




