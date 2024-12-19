import pandas as pd
import numpy as np

# Навчальні дані про погодні умови та проведення матчів
data = [
    ['Sunny', 'High', 'Weak', 'Yes'],
    ['Sunny', 'High', 'Strong', 'No'],
    ['Overcast', 'High', 'Weak', 'Yes'],
    ['Rain', 'High', 'Weak', 'No'],
    ['Rain', 'Normal', 'Weak', 'Yes'],
    ['Rain', 'Normal', 'Strong', 'No'],
    ['Overcast', 'Normal', 'Strong', 'Yes'],
    ['Sunny', 'High', 'Weak', 'Yes'],
    ['Sunny', 'Normal', 'Strong', 'No'],
    ['Rain', 'High', 'Strong', 'No']
]

# Створення DataFrame
columns = ['Outlook', 'Humidity', 'Wind', 'Play']
df = pd.DataFrame(data, columns=columns)

def naive_bayes_match_prediction(outlook, humidity, wind, dataframe):
    # Загальна кількість записів
    total_records = len(dataframe)
    
    # Ймовірність проведення матчу
    p_play = len(dataframe[dataframe['Play'] == 'Yes']) / total_records
    p_no_play = len(dataframe[dataframe['Play'] == 'No']) / total_records
    
    # Умовні ймовірності для кожної ознаки
    p_outlook_play = len(dataframe[(dataframe['Outlook'] == outlook) & (dataframe['Play'] == 'Yes')]) / len(dataframe[dataframe['Play'] == 'Yes'])
    p_humidity_play = len(dataframe[(dataframe['Humidity'] == humidity) & (dataframe['Play'] == 'Yes')]) / len(dataframe[dataframe['Play'] == 'Yes'])
    p_wind_play = len(dataframe[(dataframe['Wind'] == wind) & (dataframe['Play'] == 'Yes')]) / len(dataframe[dataframe['Play'] == 'Yes'])
    
    p_outlook_no_play = len(dataframe[(dataframe['Outlook'] == outlook) & (dataframe['Play'] == 'No')]) / len(dataframe[dataframe['Play'] == 'No'])
    p_humidity_no_play = len(dataframe[(dataframe['Humidity'] == humidity) & (dataframe['Play'] == 'No')]) / len(dataframe[dataframe['Play'] == 'No'])
    p_wind_no_play = len(dataframe[(dataframe['Wind'] == wind) & (dataframe['Play'] == 'No')]) / len(dataframe[dataframe['Play'] == 'No'])
    
    # Розрахунок апостеріорних ймовірностей
    p_match_play = p_play * p_outlook_play * p_humidity_play * p_wind_play
    p_match_no_play = p_no_play * p_outlook_no_play * p_humidity_no_play * p_wind_no_play
    
    # Нормалізація ймовірностей
    total_prob = p_match_play + p_match_no_play
    p_match_play_normalized = p_match_play / total_prob
    p_match_no_play_normalized = p_match_no_play / total_prob
    
    return {
        'Ймовірність проведення матчу': p_match_play_normalized,
        'Ймовірність скасування матчу': p_match_no_play_normalized
    }

result = naive_bayes_match_prediction('Rain', 'High', 'Strong', df)
print("Результати прогнозування:")
for key, value in result.items():
    print(f"{key}: {value * 100:.2f}%")