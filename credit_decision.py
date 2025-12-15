"""
Программа для принятия решения о выдаче кредита
Использует машинное обучение для предсказания кредитного риска
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import os


class CreditDecisionModel:
    """Класс для модели принятия решения о кредите"""
    
    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.feature_columns = None
        
    def load_data(self, file_path='german_credit_data.csv'):
        """Загрузка данных из CSV файла"""
        print("Загрузка данных...")
        df = pd.read_csv(file_path)
        print(f"Загружено {len(df)} записей")
        return df
    
    def preprocess_data(self, df):
        """Предобработка данных"""
        print("Предобработка данных...")
        
        # Удаляем первый столбец (индекс)
        if df.columns[0] == 'Unnamed: 0' or df.columns[0] == '':
            df = df.drop(df.columns[0], axis=1)
        
        # Копируем данные
        data = df.copy()
        
        # Заполняем пропущенные значения
        data['Saving accounts'] = data['Saving accounts'].fillna('none')
        data['Checking account'] = data['Checking account'].fillna('none')
        
        # Разделяем на признаки и целевую переменную
        X = data.drop('Risk', axis=1)
        y = data['Risk']
        
        # Кодируем категориальные переменные
        categorical_columns = ['Sex', 'Housing', 'Saving accounts', 'Checking account', 'Purpose']
        
        for col in categorical_columns:
            if col in X.columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.label_encoders[col] = le
        
        self.feature_columns = X.columns.tolist()
        
        # Кодируем целевую переменную
        y_encoded = (y == 'bad').astype(int)  # 1 = bad (риск), 0 = good (безопасно)
        
        return X, y_encoded
    
    def train(self, X, y):
        """Обучение модели"""
        print("Обучение модели...")
        
        # Разделяем на обучающую и тестовую выборки
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Создаем и обучаем модель Random Forest
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        
        self.model.fit(X_train, y_train)
        
        # Оценка модели
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\nТочность модели на тестовой выборке: {accuracy:.2%}")
        print("\nОтчет о классификации:")
        print(classification_report(y_test, y_pred, target_names=['Good (Безопасно)', 'Bad (Риск)']))
        
        return self.model
    
    def predict(self, person_data):
        """Предсказание для нового человека"""
        if self.model is None:
            raise ValueError("Модель не обучена! Сначала вызовите метод train()")
        
        # Создаем DataFrame из входных данных
        df = pd.DataFrame([person_data])
        
        # Кодируем категориальные переменные
        for col in self.label_encoders:
            if col in df.columns:
                # Обрабатываем новые значения
                try:
                    df[col] = self.label_encoders[col].transform(df[col].astype(str))
                except ValueError:
                    # Если значение новое, используем самое частое значение
                    df[col] = self.label_encoders[col].transform([self.label_encoders[col].classes_[0]])[0]
        
        # Убеждаемся, что порядок столбцов правильный
        df = df[self.feature_columns]
        
        # Предсказание
        prediction = self.model.predict(df)[0]
        probability = self.model.predict_proba(df)[0]
        
        return prediction, probability
    
    def save_model(self, file_path='credit_model.pkl'):
        """Сохранение модели"""
        model_data = {
            'model': self.model,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns
        }
        with open(file_path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Модель сохранена в {file_path}")
    
    def load_model(self, file_path='credit_model.pkl'):
        """Загрузка модели"""
        with open(file_path, 'rb') as f:
            model_data = pickle.load(f)
        self.model = model_data['model']
        self.label_encoders = model_data['label_encoders']
        self.feature_columns = model_data['feature_columns']
        print(f"Модель загружена из {file_path}")


def get_person_input():
    """Получение данных о человеке от пользователя"""
    print("\n" + "="*50)
    print("Введите данные о заемщике:")
    print("="*50)
    
    person = {}
    
    try:
        person['Age'] = int(input("Возраст: "))
        person['Sex'] = input("Пол (male/female): ").lower()
        person['Job'] = int(input("Работа (0=безработный, 1=неквалифицированная, 2=квалифицированная, 3=высококвалифицированная): "))
        person['Housing'] = input("Жилье (own/rent/free): ").lower()
        
        saving = input("Сберегательный счет (little/moderate/quite rich/rich/none): ").lower()
        person['Saving accounts'] = saving if saving else 'none'
        
        checking = input("Текущий счет (little/moderate/rich/none): ").lower()
        person['Checking account'] = checking if checking else 'none'
        
        person['Credit amount'] = float(input("Сумма кредита: "))
        person['Duration'] = int(input("Срок кредита (месяцы): "))
        
        purpose = input("Цель кредита (car/radio/TV/furniture/equipment/business/education/repairs/vacation/others/domestic appliances): ").lower()
        person['Purpose'] = purpose
        
        return person
    except ValueError as e:
        print(f"Ошибка ввода: {e}")
        return None


def main():
    """Основная функция"""
    print("="*60)
    print("ПРОГРАММА ПРИНЯТИЯ РЕШЕНИЯ О ВЫДАЧЕ КРЕДИТА")
    print("="*60)
    
    model = CreditDecisionModel()
    
    # Проверяем, существует ли сохраненная модель
    if os.path.exists('credit_model.pkl'):
        load = input("\nНайдена сохраненная модель. Загрузить её? (y/n): ").lower()
        if load == 'y':
            try:
                model.load_model()
            except Exception as e:
                print(f"Ошибка загрузки модели: {e}")
                print("Обучаем новую модель...")
                df = model.load_data()
                X, y = model.preprocess_data(df)
                model.train(X, y)
                model.save_model()
        else:
            df = model.load_data()
            X, y = model.preprocess_data(df)
            model.train(X, y)
            model.save_model()
    else:
        # Обучаем модель
        df = model.load_data()
        X, y = model.preprocess_data(df)
        model.train(X, y)
        model.save_model()
    
    # Основной цикл предсказаний
    while True:
        person = get_person_input()
        
        if person is None:
            continue
        
        try:
            prediction, probability = model.predict(person)
            
            print("\n" + "="*60)
            print("РЕЗУЛЬТАТ ПРЕДСКАЗАНИЯ:")
            print("="*60)
            
            if prediction == 0:
                decision = "ОДОБРИТЬ КРЕДИТ"
                risk_level = "НИЗКИЙ РИСК"
                emoji = "✓"
            else:
                decision = "ОТКАЗАТЬ В КРЕДИТЕ"
                risk_level = "ВЫСОКИЙ РИСК"
                emoji = "✗"
            
            print(f"\nРешение: {emoji} {decision}")
            print(f"Уровень риска: {risk_level}")
            print(f"\nВероятность низкого риска (good): {probability[0]:.2%}")
            print(f"Вероятность высокого риска (bad): {probability[1]:.2%}")
            
        except Exception as e:
            print(f"Ошибка при предсказании: {e}")
        
        continue_choice = input("\nПродолжить? (y/n): ").lower()
        if continue_choice != 'y':
            break
    
    print("\nСпасибо за использование программы!")


if __name__ == "__main__":
    main()

