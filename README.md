# Программа для принятия решения о выдаче кредита

Эта программа использует машинное обучение для предсказания кредитного риска на основе данных о заемщике.

## Установка

1. Установите необходимые библиотеки:
```bash
pip install -r requirements.txt
```

## Использование

### Способ 1: Интерактивный режим (рекомендуется)

Просто запустите программу:
```bash
python credit_decision.py
```

Программа:
1. При первом запуске автоматически обучит модель на данных из `german_credit_data.csv`
2. Сохранит обученную модель в файл `credit_model.pkl`
3. Попросит ввести данные о заемщике
4. Выдаст решение: одобрить или отказать в кредите

### Пример ввода данных:

```
Возраст: 30
Пол (male/female): male
Работа (0=безработный, 1=неквалифицированная, 2=квалифицированная, 3=высококвалифицированная): 2
Жилье (own/rent/free): own
Сберегательный счет (little/moderate/quite rich/rich/none): moderate
Текущий счет (little/moderate/rich/none): moderate
Сумма кредита: 5000
Срок кредита (месяцы): 24
Цель кредита (car/radio/TV/furniture/equipment/business/education/repairs/vacation/others/domestic appliances): car
```

### Способ 2: Использование в своем коде

```python
from credit_decision import CreditDecisionModel

# Создаем модель
model = CreditDecisionModel()

# Загружаем сохраненную модель (если она уже обучена)
model.load_model('credit_model.pkl')

# Или обучаем новую модель
# df = model.load_data('german_credit_data.csv')
# X, y = model.preprocess_data(df)
# model.train(X, y)
# model.save_model('credit_model.pkl')

# Данные о заемщике
person = {
    'Age': 30,
    'Sex': 'male',
    'Job': 2,
    'Housing': 'own',
    'Saving accounts': 'moderate',
    'Checking account': 'moderate',
    'Credit amount': 5000,
    'Duration': 24,
    'Purpose': 'car'
}

# Предсказание
prediction, probability = model.predict(person)

if prediction == 0:
    print("✓ Кредит одобрен (низкий риск)")
else:
    print("✗ Кредит отклонен (высокий риск)")

print(f"Вероятность низкого риска: {probability[0]:.2%}")
print(f"Вероятность высокого риска: {probability[1]:.2%}")
```

## Описание полей

- **Age** - Возраст заемщика (число)
- **Sex** - Пол: `male` или `female`
- **Job** - Тип работы:
  - `0` - безработный
  - `1` - неквалифицированная работа
  - `2` - квалифицированная работа
  - `3` - высококвалифицированная работа
- **Housing** - Тип жилья: `own` (собственное), `rent` (аренда), `free` (бесплатное)
- **Saving accounts** - Сберегательный счет: `little`, `moderate`, `quite rich`, `rich`, `none`
- **Checking account** - Текущий счет: `little`, `moderate`, `rich`, `none`
- **Credit amount** - Сумма кредита (число)
- **Duration** - Срок кредита в месяцах (число)
- **Purpose** - Цель кредита: `car`, `radio/TV`, `furniture/equipment`, `business`, `education`, `repairs`, `vacation/others`, `domestic appliances`

## Результат

Программа выдает:
- **Решение**: одобрить или отказать в кредите
- **Уровень риска**: низкий или высокий
- **Вероятности**: процент вероятности для каждого класса риска

