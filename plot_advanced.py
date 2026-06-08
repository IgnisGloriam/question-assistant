import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from math import pi

# Настройка шрифтов
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 12

# Загрузка данных
try:
    df = pd.read_csv("results_xquad.csv")
except FileNotFoundError:
    print("Файл results_xquad.csv не найден!")
    exit()

# Приводим имена моделей к красивому виду (например, оставляем только '3b' и '1.5b')
df['model_short'] = df['model_name'].apply(lambda x: "Qwen " + [part for part in x.split('-') if 'b' in part.lower()][0].upper())
models = df['model_short'].unique()

# 5 основных метрик из курсовой
metrics = ['format_adherence', 'semantic_sim', 'distractor_distinctness', 'ragas_faithfulness', 'rouge_l']
metrics_labels = ['Формат\n(Format)', 'Семантика\n(Sem Sim)', 'Дистракторы\n(Distinctness)', 'Фактология\n(Faithfulness)', 'Лексика\n(ROUGE-L)']

# ================= 1. РАДАРНАЯ ДИАГРАММА (Комплексный профиль) =================
agg_df = df.groupby('model_short')[metrics].mean().reset_index()

N = len(metrics)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]  # Замыкаем круг

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
ax.set_theta_offset(pi / 2)
ax.set_theta_direction(-1)
plt.xticks(angles[:-1], metrics_labels, size=11)
ax.set_rlabel_position(0)
plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2","0.4","0.6","0.8","1.0"], color="grey", size=10)
plt.ylim(0, 1.1)

colors = ['#4C72B0', '#C44E52']
for i, model in enumerate(models):
    values = agg_df[agg_df['model_short'] == model][metrics].values.flatten().tolist()
    values += values[:1] # Замыкаем круг
    ax.plot(angles, values, linewidth=2, linestyle='solid', label=model, color=colors[i % len(colors)])
    ax.fill(angles, values, color=colors[i % len(colors)], alpha=0.1)

plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.title("Комплексный профиль моделей по 5 метрикам", size=14, y=1.1)
plt.tight_layout()
plt.savefig('radar_metrics.pdf')
plt.close()

# ================= 2. BOXPLOT (Стабильность генерации) =================
# Выберем две самые показательные метрики для распределения
box_metrics = ['semantic_sim', 'ragas_faithfulness']
box_labels = ['Семантическое сходство', 'Фактологическая точность']

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for i, metric in enumerate(box_metrics):
    sns.boxplot(x='model_short', y=metric, data=df, ax=axes[i], palette='Set2', width=0.5)
    axes[i].set_title(box_labels[i])
    axes[i].set_ylabel('Значение (0-1)')
    axes[i].set_xlabel('')
    axes[i].grid(axis='y', linestyle='--', alpha=0.7)

plt.suptitle("Распределение метрик качества (Стабильность генерации)", fontsize=14)
plt.tight_layout()
plt.savefig('boxplot_stability.pdf')
plt.close()

# ================= 3. SCATTER PLOT (Иллюстрация парадокса) =================
# Показываем связь между Семантикой и Лексикой (ROUGE-L)
fig, ax = plt.subplots(figsize=(8, 6))

# Берем лучшую модель для демонстрации парадокса
best_model = models[0] 
subset = df[df['model_short'] == best_model]

sns.scatterplot(data=subset, x='semantic_sim', y='rouge_l', alpha=0.7, s=80, color='#55A868', edgecolor='k')

# Выделяем "Зону парадокса" (Высокая семантика, низкая лексика)
ax.axvspan(0.7, 1.0, ymin=0, ymax=0.3, color='red', alpha=0.1, label='Зона парадокса')

ax.set_title(f"Связь семантического и лексического сходства ({best_model})")
ax.set_xlabel("Семантическое сходство (Semantic Sim)")
ax.set_ylabel("Лексическое совпадение (ROUGE-L)")
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend()

plt.tight_layout()
plt.savefig('scatter_paradox.pdf')
plt.close()

print("✅ Все графики успешно сгенерированы (radar_metrics.pdf, boxplot_stability.pdf, scatter_paradox.pdf)")