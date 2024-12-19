import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime

def load_and_prepare_data():
    """Load and prepare the train pricing data"""
    # Read the dataset
    df = pd.read_csv('renfe_small.csv', parse_dates=['insert_date', 'start_date', 'end_date'])
    
    # Remove rows with missing prices
    df = df.dropna(subset=['price'])
    
    # Create duration in hours
    df['duration'] = (df['end_date'] - df['start_date']).dt.total_seconds() / 3600
    
    # Create binary features
    df['is_premium'] = (df['train_class'] == 'Preferente').astype(int)
    df['is_flexible'] = (df['fare'].str.contains('Flexible')).astype(int)
    
    return df

def analyze_price_distribution(df):
    """Analyze the price distribution and its relationship with other variables"""
    results = {}
    
    # Overall price statistics
    mean_price = df['price'].mean()
    std_error = stats.sem(df['price'])
    confidence_interval = stats.norm.interval(confidence=0.95, loc=mean_price, scale=std_error)
    
    results['overall'] = {
        'mean': mean_price,
        'median': df['price'].median(),
        'std': df['price'].std(),
        'confidence_interval': confidence_interval
    }
    
    # Price by train class
    class_stats = df.groupby('train_class')['price'].agg(['mean', 'std', 'count']).round(2)
    results['by_class'] = class_stats.to_dict('index')
    
    # Price by train type
    type_stats = df.groupby('train_type')['price'].agg(['mean', 'std', 'count']).round(2)
    results['by_train_type'] = type_stats.to_dict('index')
    
    # Correlation with duration
    results['duration_correlation'] = stats.pearsonr(df['duration'], df['price'])
    
    return results

def analyze_routes(df):
    """Analyze price patterns for different routes"""
    route_stats = df.groupby(['origin', 'destination']).agg({
        'price': ['mean', 'std', 'count'],
        'duration': 'mean'
    }).round(2)
    
    # Calculate price per hour for each route
    route_stats['price_per_hour'] = (
        route_stats[('price', 'mean')] / route_stats[('duration', 'mean')]
    ).round(2)
    
    return route_stats

def print_detailed_stats(df):
    """Print additional detailed statistics"""
    print("\n5. Додаткова статистика:")
    
    # Price distribution by fare type
    print("\nСередні ціни за типами тарифів:")
    print(df.groupby('fare')['price'].agg(['mean', 'count']).round(2))
    
    # Busiest routes
    print("\nНайпопулярніші маршрути:")
    route_counts = df.groupby(['origin', 'destination']).size().sort_values(ascending=False).head()
    print(route_counts)
    
    # Price ranges
    print("\nДіапазони цін:")
    print(f"Мінімальна ціна: {df['price'].min():.2f} €")
    print(f"Максимальна ціна: {df['price'].max():.2f} €")
    print(f"Діапазон цін: {df['price'].max() - df['price'].min():.2f} €")

def main():
    # Load and prepare data
    print("Завантаження та підготовка даних...")
    df = load_and_prepare_data()
    
    # Analyze price distribution
    print("\nАналіз розподілу цін...")
    price_analysis = analyze_price_distribution(df)
    
    # Analyze routes
    print("\nАналіз маршрутів...")
    route_analysis = analyze_routes(df)
    
    # Print results
    print("\nРЕЗУЛЬТАТИ АНАЛІЗУ:")
    print("\n1. Загальна статистика цін:")
    print(f"Середня ціна: {price_analysis['overall']['mean']:.2f} €")
    print(f"Медіанна ціна: {price_analysis['overall']['median']:.2f} €")
    print(f"Стандартне відхилення: {price_analysis['overall']['std']:.2f} €")
    print(f"95% довірчий інтервал: [{price_analysis['overall']['confidence_interval'][0]:.2f}, "
          f"{price_analysis['overall']['confidence_interval'][1]:.2f}] €")
    
    print("\n2. Ціни за класами квитків:")
    for train_class, stats in price_analysis['by_class'].items():
        print(f"\n{train_class}:")
        print(f"Середня ціна: {stats['mean']:.2f} €")
        print(f"Стандартне відхилення: {stats['std']:.2f} €")
        print(f"Кількість квитків: {stats['count']}")
    
    print("\n3. Кореляція між тривалістю та ціною:")
    correlation, p_value = price_analysis['duration_correlation']
    print(f"Коефіцієнт кореляції: {correlation:.3f}")
    print(f"P-значення: {p_value:.3f}")
    
    print("\n4. Статистика за маршрутами:")
    print(route_analysis.to_string())
    
    # Print additional detailed statistics
    print_detailed_stats(df)

if __name__ == "__main__":
    main()