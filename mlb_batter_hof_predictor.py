#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFE
from mlxtend.classifier import StackingClassifier
import numpy as np


def load_data():
    """Return main dataframes from CSVs to read in the Lahman database"""
    df_people = pd.read_csv('./baseballdatabank-2023.1/core/People.csv')
    df_batting = pd.read_csv('./baseballdatabank-2023.1/core/Batting.csv')
    df_hall_of_fame = pd.read_csv(
        './baseballdatabank-2023.1/contrib/HallOfFame.csv')

    return df_people, df_batting, df_hall_of_fame

def hall_of_fame_preprocessing(df_hall_of_fame):
    """Make sure we isolate the last ballot of any batter"""
    df_hall_of_fame_condensed = df_hall_of_fame[['playerID','yearID','inducted']]
    df_hall_of_fame_condensed = df_hall_of_fame_condensed.sort_values('yearID',ascending=True)
    df_hall_of_fame_condensed = df_hall_of_fame_condensed.drop_duplicates(subset='playerID', keep='last')
    return df_hall_of_fame_condensed

def engineer_features(df_hof_batters_full):
    """Calculate OBP and SLG based on features in dataframe"""
    df_hof_batters_expanded = df_hof_batters_full
    obp_list = []
    slg_list = []
    for index, row in df_hof_batters_expanded.iterrows():
        if row['AB'] > 0:
            obp_value = (row['H'] + row['BB'] + row['HBP']) / \
                (row['AB'] + row['BB'] + row['HBP'] + row['SF'])
            slg_value = (row['H'] + row['2B'] + (row['3B']
                                                 * 2) + (row['HR'] * 3)) / row['AB']
            obp_list.append(obp_value)
            slg_list.append(slg_value)
        else:
            obp_list.append(0)
            slg_list.append(0)

    df_hof_batters_expanded['OBP'] = obp_list
    df_hof_batters_expanded['SLG'] = slg_list

    return df_hof_batters_expanded


def split_data(df_hof_batters_even_classes):
    """Define X and Y values to create train test split"""

    X = df_hof_batters_even_classes.drop('inducted', axis=1)
    y = df_hof_batters_even_classes['inducted']

    # We will use a StandardScaler for this data.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Create the train test split.
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def grid_search_report(X_train, y_train, X_test, y_test):
    """Print out a classification report for three different
    ML models to see which ones fair the best, running each
    through grid search for optimization.
    """

    # Define models and their parameter grids
    models_params = {
        'logistic_regression': {
            'model': LogisticRegression(),
            'params': {'C': [0.1, 1, 10], 'penalty': ['l1', 'l2']}
        },
        'knn': {
            'model': KNeighborsClassifier(),
            'params': {'n_neighbors': [3, 5, 7, 9]}
        },
        'decision_tree': {
            'model': DecisionTreeClassifier(),
            'params': {'max_depth': [3, 5, 7, 9]}
        }
    }

    # Run GridSearchCV on each model, print the report
    for model_name, mp in models_params.items():
        clf = GridSearchCV(mp['model'], mp['params'],
                           cv=5, return_train_score=False)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        print(f"Model: {model_name}")
        print(f"Best Parameters: {clf.best_params_}")
        print(f"Accuracy Score: {accuracy_score(y_test, y_pred)}\n")
        print(classification_report(y_test, y_pred))


def stacked_classifier_report(X_train, y_train, X_test, y_test):
    """Use a stacking classifier based on the models
    from the above function, and print out a
    clasification report to see which one performs better.
    """

    m = StackingClassifier(
        classifiers=[
            LogisticRegression(C=0.1, penalty='l2'),
            KNeighborsClassifier(n_neighbors=9),
            DecisionTreeClassifier(max_depth=5)
        ],
        use_probas=True,
        meta_classifier=LogisticRegression()
    )

    m.fit(X_train, y_train)
    y_pred = m.predict(X_test)

    print(f"Mixed Model Accuracy Score: {accuracy_score(y_test, y_pred)}\n")
    print(classification_report(y_test, y_pred))


# Load the dataframes first.
df_people, df_batting, df_hall_of_fame = load_data()

# Make sure we group batting data by the playerID for future calculations,
# then merge with the Hall of Fame table.
df_batting_group = df_batting.groupby(['playerID']).sum().reset_index()

df_hall_of_fame_condensed = hall_of_fame_preprocessing(df_hall_of_fame)

df_hof_batters = pd.merge(
    df_hall_of_fame_condensed[['playerID', 'inducted']], df_batting_group, on="playerID")

# Filter down to key measurements.
df_hof_batters = df_hof_batters[['playerID',
                                 'AB',
                                 'R',
                                 'H',
                                 '2B',
                                 '3B',
                                 'HR',
                                 'RBI',
                                 'SB',
                                 'CS',
                                 'BB',
                                 'SO',
                                 'IBB',
                                 'HBP',
                                 'SH',
                                 'SF',
                                 'GIDP',
                                 'inducted']]

df_hof_batters_sum = df_hof_batters.groupby('playerID').sum().reset_index()

df_hall_of_fame_no_dups = df_hall_of_fame[[
    'playerID', 'inducted']].drop_duplicates()

df_hof_batters_full = pd.merge(df_hof_batters_sum, df_hall_of_fame_no_dups[[
                               'playerID', 'inducted']], on="playerID", how='inner')

df_hof_batters_expanded = engineer_features(df_hof_batters_full)

df_hof_batters_even_classes = df_hof_batters_expanded.groupby(
    'inducted').apply(lambda x: x.sample(n=216)).reset_index(drop=True)

df_hof_batters_even_classes = df_hof_batters_even_classes[[
    'OBP', 'SLG', 'SO', 'SH', 'SF', 'inducted']]

X_train, X_test, y_train, y_test = split_data(df_hof_batters_even_classes)

grid_search_report(X_train, y_train, X_test, y_test)
stacked_classifier_report(X_train, y_train, X_test, y_test)
