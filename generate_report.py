"""
Скрипт для генерации детального отчета с графиками и анализом
для урока по машинному обучению
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import warnings
warnings.filterwarnings('ignore')

# Настройка для русского языка в графиках
plt.rcParams['font.family'] = 'DejaVu Sans'
sns.set_style("whitegrid")
sns.set_palette("husl")

class CreditAnalysisReport:
    """Класс для создания детального отчета"""
    
    def __init__(self):
        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.label_encoders = {}
        
    def load_and_prepare_data(self):
        """Загрузка и подготовка данных"""
        print("Загрузка данных...")
        self.df = pd.read_csv('german_credit_data.csv')
        
        # Удаляем первый столбец если это индекс
        if self.df.columns[0] == 'Unnamed: 0' or self.df.columns[0] == '':
            self.df = self.df.drop(self.df.columns[0], axis=1)
        
        # Заполняем пропущенные значения
        self.df['Saving accounts'] = self.df['Saving accounts'].fillna('none')
        self.df['Checking account'] = self.df['Checking account'].fillna('none')
        
        # Подготовка признаков
        X = self.df.drop('Risk', axis=1).copy()
        y = self.df['Risk'].copy()
        
        # Кодирование категориальных переменных
        categorical_columns = ['Sex', 'Housing', 'Saving accounts', 'Checking account', 'Purpose']
        
        for col in categorical_columns:
            if col in X.columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.label_encoders[col] = le
        
        # Кодирование целевой переменной
        y_encoded = (y == 'bad').astype(int)
        
        self.X = X
        self.y = y_encoded
        
        # Разделение на обучающую и тестовую выборки
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        print(f"Данные загружены: {len(self.df)} записей")
        print(f"Обучающая выборка: {len(self.X_train)} записей")
        print(f"Тестовая выборка: {len(self.X_test)} записей")
        
    def train_model(self):
        """Обучение модели"""
        print("\nОбучение модели Random Forest...")
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        self.model.fit(self.X_train, self.y_train)
        print("Модель обучена!")
        
    def create_all_visualizations(self):
        """Создание всех визуализаций"""
        print("\nСоздание графиков и диаграмм...")
        
        # Создаем папку для графиков
        import os
        os.makedirs('report_images', exist_ok=True)
        
        # 1. Распределение целевой переменной
        self.plot_target_distribution()
        
        # 2. Распределение возраста
        self.plot_age_distribution()
        
        # 3. Распределение суммы кредита
        self.plot_credit_amount_distribution()
        
        # 4. Корреляционная матрица
        self.plot_correlation_matrix()
        
        # 5. Важность признаков
        self.plot_feature_importance()
        
        # 6. Матрица ошибок
        self.plot_confusion_matrix()
        
        # 7. ROC кривая
        self.plot_roc_curve()
        
        # 8. Precision-Recall кривая
        self.plot_precision_recall_curve()
        
        # 9. Анализ по категориям
        self.plot_categorical_analysis()
        
        # 10. Сравнение распределений для хороших и плохих кредитов
        self.plot_comparison_distributions()
        
        print("\nВсе графики сохранены в папку 'report_images'")
        
    def plot_target_distribution(self):
        """Распределение целевой переменной"""
        fig, ax = plt.subplots(figsize=(10, 6))
        risk_counts = self.df['Risk'].value_counts()
        colors = ['#2ecc71', '#e74c3c']
        bars = ax.bar(risk_counts.index, risk_counts.values, color=colors, alpha=0.7, edgecolor='black')
        
        ax.set_xlabel('Risk Category', fontsize=12, fontweight='bold')
        ax.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax.set_title('Distribution of Credit Risk (Target Variable)', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Добавляем значения на столбцы
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}\n({height/len(self.df)*100:.1f}%)',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('report_images/01_target_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_age_distribution(self):
        """Распределение возраста"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Гистограмма
        axes[0].hist(self.df['Age'], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('Age', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
        axes[0].set_title('Age Distribution (Histogram)', fontsize=13, fontweight='bold')
        axes[0].grid(axis='y', alpha=0.3)
        axes[0].axvline(self.df['Age'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {self.df["Age"].mean():.1f}')
        axes[0].axvline(self.df['Age'].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {self.df["Age"].median():.1f}')
        axes[0].legend()
        
        # Box plot по риску
        risk_data = [self.df[self.df['Risk'] == 'good']['Age'], 
                     self.df[self.df['Risk'] == 'bad']['Age']]
        bp = axes[1].boxplot(risk_data, labels=['Good', 'Bad'], patch_artist=True)
        bp['boxes'][0].set_facecolor('#2ecc71')
        bp['boxes'][1].set_facecolor('#e74c3c')
        axes[1].set_ylabel('Age', fontsize=12, fontweight='bold')
        axes[1].set_title('Age Distribution by Risk Category', fontsize=13, fontweight='bold')
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('report_images/02_age_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_credit_amount_distribution(self):
        """Распределение суммы кредита"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Гистограмма
        axes[0].hist(self.df['Credit amount'], bins=50, color='coral', edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('Credit Amount', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
        axes[0].set_title('Credit Amount Distribution', fontsize=13, fontweight='bold')
        axes[0].grid(axis='y', alpha=0.3)
        
        # Логарифмическая шкала
        axes[1].hist(np.log1p(self.df['Credit amount']), bins=50, color='coral', edgecolor='black', alpha=0.7)
        axes[1].set_xlabel('Log(Credit Amount + 1)', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Frequency', fontsize=12, fontweight='bold')
        axes[1].set_title('Credit Amount Distribution (Log Scale)', fontsize=13, fontweight='bold')
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('report_images/03_credit_amount_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_correlation_matrix(self):
        """Корреляционная матрица"""
        # Выбираем только числовые признаки
        numeric_cols = ['Age', 'Job', 'Credit amount', 'Duration']
        corr_data = self.df[numeric_cols + ['Risk']].copy()
        corr_data['Risk_encoded'] = (corr_data['Risk'] == 'bad').astype(int)
        corr_matrix = corr_data[numeric_cols + ['Risk_encoded']].corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                   square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
        ax.set_title('Correlation Matrix of Numerical Features', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig('report_images/04_correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_feature_importance(self):
        """Важность признаков"""
        feature_importance = pd.DataFrame({
            'feature': self.X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        bars = ax.barh(feature_importance['feature'], feature_importance['importance'], 
                       color='steelblue', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Importance', fontsize=12, fontweight='bold')
        ax.set_ylabel('Feature', fontsize=12, fontweight='bold')
        ax.set_title('Feature Importance (Random Forest)', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # Добавляем значения
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2,
                   f'{width:.3f}', ha='left', va='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig('report_images/05_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_confusion_matrix(self):
        """Матрица ошибок"""
        y_pred = self.model.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Good', 'Bad'], yticklabels=['Good', 'Bad'])
        ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
        ax.set_ylabel('Actual', fontsize=12, fontweight='bold')
        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('report_images/06_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_roc_curve(self):
        """ROC кривая"""
        y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(self.y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        ax.set_title('ROC Curve', fontsize=14, fontweight='bold')
        ax.legend(loc="lower right", fontsize=11)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('report_images/07_roc_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_precision_recall_curve(self):
        """Precision-Recall кривая"""
        y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        precision, recall, thresholds = precision_recall_curve(self.y_test, y_pred_proba)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot(recall, precision, color='blue', lw=2)
        ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
        ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
        ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('report_images/08_precision_recall_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_categorical_analysis(self):
        """Анализ категориальных переменных"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Sex
        sex_risk = pd.crosstab(self.df['Sex'], self.df['Risk'])
        sex_risk.plot(kind='bar', ax=axes[0, 0], color=['#2ecc71', '#e74c3c'], alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('Risk Distribution by Sex', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Sex', fontsize=11)
        axes[0, 0].set_ylabel('Count', fontsize=11)
        axes[0, 0].legend(title='Risk')
        axes[0, 0].tick_params(axis='x', rotation=0)
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # Housing
        housing_risk = pd.crosstab(self.df['Housing'], self.df['Risk'])
        housing_risk.plot(kind='bar', ax=axes[0, 1], color=['#2ecc71', '#e74c3c'], alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('Risk Distribution by Housing', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Housing', fontsize=11)
        axes[0, 1].set_ylabel('Count', fontsize=11)
        axes[0, 1].legend(title='Risk')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # Saving accounts
        saving_risk = pd.crosstab(self.df['Saving accounts'], self.df['Risk'])
        saving_risk.plot(kind='bar', ax=axes[1, 0], color=['#2ecc71', '#e74c3c'], alpha=0.7, edgecolor='black')
        axes[1, 0].set_title('Risk Distribution by Saving Accounts', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Saving Accounts', fontsize=11)
        axes[1, 0].set_ylabel('Count', fontsize=11)
        axes[1, 0].legend(title='Risk')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(axis='y', alpha=0.3)
        
        # Purpose
        purpose_risk = pd.crosstab(self.df['Purpose'], self.df['Risk'])
        purpose_risk.plot(kind='bar', ax=axes[1, 1], color=['#2ecc71', '#e74c3c'], alpha=0.7, edgecolor='black')
        axes[1, 1].set_title('Risk Distribution by Purpose', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Purpose', fontsize=11)
        axes[1, 1].set_ylabel('Count', fontsize=11)
        axes[1, 1].legend(title='Risk')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('report_images/09_categorical_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_comparison_distributions(self):
        """Сравнение распределений для хороших и плохих кредитов"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        good_risk = self.df[self.df['Risk'] == 'good']
        bad_risk = self.df[self.df['Risk'] == 'bad']
        
        # Credit Amount
        axes[0, 0].hist(good_risk['Credit amount'], bins=30, alpha=0.6, label='Good', color='#2ecc71', edgecolor='black')
        axes[0, 0].hist(bad_risk['Credit amount'], bins=30, alpha=0.6, label='Bad', color='#e74c3c', edgecolor='black')
        axes[0, 0].set_xlabel('Credit Amount', fontsize=12, fontweight='bold')
        axes[0, 0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
        axes[0, 0].set_title('Credit Amount Distribution by Risk', fontsize=13, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        # Duration
        axes[0, 1].hist(good_risk['Duration'], bins=30, alpha=0.6, label='Good', color='#2ecc71', edgecolor='black')
        axes[0, 1].hist(bad_risk['Duration'], bins=30, alpha=0.6, label='Bad', color='#e74c3c', edgecolor='black')
        axes[0, 1].set_xlabel('Duration (months)', fontsize=12, fontweight='bold')
        axes[0, 1].set_ylabel('Frequency', fontsize=12, fontweight='bold')
        axes[0, 1].set_title('Duration Distribution by Risk', fontsize=13, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
        
        # Age
        axes[1, 0].hist(good_risk['Age'], bins=30, alpha=0.6, label='Good', color='#2ecc71', edgecolor='black')
        axes[1, 0].hist(bad_risk['Age'], bins=30, alpha=0.6, label='Bad', color='#e74c3c', edgecolor='black')
        axes[1, 0].set_xlabel('Age', fontsize=12, fontweight='bold')
        axes[1, 0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
        axes[1, 0].set_title('Age Distribution by Risk', fontsize=13, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)
        
        # Job
        job_risk = pd.crosstab(self.df['Job'], self.df['Risk'])
        job_risk.plot(kind='bar', ax=axes[1, 1], color=['#2ecc71', '#e74c3c'], alpha=0.7, edgecolor='black')
        axes[1, 1].set_title('Risk Distribution by Job Type', fontsize=13, fontweight='bold')
        axes[1, 1].set_xlabel('Job Type', fontsize=12, fontweight='bold')
        axes[1, 1].set_ylabel('Count', fontsize=12, fontweight='bold')
        axes[1, 1].legend(title='Risk')
        axes[1, 1].tick_params(axis='x', rotation=0)
        axes[1, 1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('report_images/10_comparison_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def generate_metrics_report(self):
        """Генерация отчета с метриками"""
        y_pred = self.model.predict(self.X_test)
        y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        
        accuracy = accuracy_score(self.y_test, y_pred)
        cm = confusion_matrix(self.y_test, y_pred)
        fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        report = classification_report(self.y_test, y_pred, target_names=['Good', 'Bad'], output_dict=True)
        
        metrics = {
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'feature_importance': dict(zip(self.X.columns, self.model.feature_importances_))
        }
        
        return metrics


def main():
    """Основная функция"""
    print("="*60)
    print("ГЕНЕРАЦИЯ ДЕТАЛЬНОГО ОТЧЕТА С ГРАФИКАМИ")
    print("="*60)
    
    analyzer = CreditAnalysisReport()
    
    # Загрузка и подготовка данных
    analyzer.load_and_prepare_data()
    
    # Обучение модели
    analyzer.train_model()
    
    # Создание всех визуализаций
    analyzer.create_all_visualizations()
    
    # Генерация метрик
    metrics = analyzer.generate_metrics_report()
    
    print("\n" + "="*60)
    print("ОСНОВНЫЕ МЕТРИКИ МОДЕЛИ:")
    print("="*60)
    print(f"Точность (Accuracy): {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    print("\nМатрица ошибок:")
    print(f"  True Negative (Good->Good):  {metrics['confusion_matrix'][0][0]}")
    print(f"  False Positive (Good->Bad): {metrics['confusion_matrix'][0][1]}")
    print(f"  False Negative (Bad->Good): {metrics['confusion_matrix'][1][0]}")
    print(f"  True Positive (Bad->Bad):   {metrics['confusion_matrix'][1][1]}")
    
    print("\n" + "="*60)
    print("Отчет успешно сгенерирован!")
    print("Все графики сохранены в папку 'report_images'")
    print("="*60)


if __name__ == "__main__":
    main()

