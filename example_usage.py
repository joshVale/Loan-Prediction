"""
Простой пример использования модели кредитного решения
"""

from credit_decision import CreditDecisionModel
import os

def example_usage():
    """Пример использования модели"""
    
    print("="*60)
    print("ПРИМЕР ИСПОЛЬЗОВАНИЯ МОДЕЛИ КРЕДИТНОГО РЕШЕНИЯ")
    print("="*60)
    
    # Создаем модель
    model = CreditDecisionModel()
    
    # Проверяем, есть ли сохраненная модель
    if os.path.exists('credit_model.pkl'):
        print("\nЗагружаем сохраненную модель...")
        model.load_model()
    else:
        print("\nОбучаем новую модель...")
        df = model.load_data('german_credit_data.csv')
        X, y = model.preprocess_data(df)
        model.train(X, y)
        model.save_model()
    
    # Пример 1: Хороший заемщик
    print("\n" + "="*60)
    print("ПРИМЕР 1: Хороший заемщик")
    print("="*60)
    
    good_borrower = {
        'Age': 35,
        'Sex': 'male',
        'Job': 2,  # квалифицированная работа
        'Housing': 'own',  # собственное жилье
        'Saving accounts': 'moderate',  # умеренные сбережения
        'Checking account': 'moderate',  # умеренный текущий счет
        'Credit amount': 3000,
        'Duration': 12,
        'Purpose': 'car'
    }
    
    prediction, probability = model.predict(good_borrower)
    
    print(f"\nДанные заемщика:")
    for key, value in good_borrower.items():
        print(f"  {key}: {value}")
    
    print(f"\nРезультат:")
    if prediction == 0:
        print("  ✓ КРЕДИТ ОДОБРЕН (низкий риск)")
    else:
        print("  ✗ КРЕДИТ ОТКЛОНЕН (высокий риск)")
    
    print(f"  Вероятность низкого риска: {probability[0]:.2%}")
    print(f"  Вероятность высокого риска: {probability[1]:.2%}")
    
    # Пример 2: Рискованный заемщик
    print("\n" + "="*60)
    print("ПРИМЕР 2: Рискованный заемщик")
    print("="*60)
    
    risky_borrower = {
        'Age': 22,
        'Sex': 'male',
        'Job': 0,  # безработный
        'Housing': 'rent',  # аренда
        'Saving accounts': 'none',  # нет сбережений
        'Checking account': 'none',  # нет текущего счета
        'Credit amount': 15000,
        'Duration': 60,
        'Purpose': 'business'
    }
    
    prediction, probability = model.predict(risky_borrower)
    
    print(f"\nДанные заемщика:")
    for key, value in risky_borrower.items():
        print(f"  {key}: {value}")
    
    print(f"\nРезультат:")
    if prediction == 0:
        print("  ✓ КРЕДИТ ОДОБРЕН (низкий риск)")
    else:
        print("  ✗ КРЕДИТ ОТКЛОНЕН (высокий риск)")
    
    print(f"  Вероятность низкого риска: {probability[0]:.2%}")
    print(f"  Вероятность высокого риска: {probability[1]:.2%}")
    
    print("\n" + "="*60)
    print("Пример завершен!")
    print("="*60)


if __name__ == "__main__":
    example_usage()

