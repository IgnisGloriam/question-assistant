import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 12

csv_file = "results_xquad.csv"
try:
    df = pd.read_csv(csv_file)
except FileNotFoundError:
    print(f"Файл {csv_file} не найден. Сначала запустите evaluate_rag.py")
    exit()

metrics_to_plot = ['format_adherence', 'semantic_sim', 'ragas_faithfulness']
agg_df = df.groupby('model_name')[metrics_to_plot].mean().reset_index()

models = agg_df['model_name'].apply(lambda x: x.split('-')[1]).tolist() # Берем просто '3b', '1.5b' для краткости
format_scores = agg_df['format_adherence'].tolist()
semsim_scores = agg_df['semantic_sim'].tolist()
faith_scores = agg_df['ragas_faithfulness'].tolist()

x = np.arange(len(models))  
width = 0.25  


fig, ax = plt.subplots(figsize=(10, 6))

rects1 = ax.bar(x - width, format_scores, width, label='Format Adherence', color='#4C72B0')
rects2 = ax.bar(x, semsim_scores, width, label='Semantic Similarity', color='#55A868')
rects3 = ax.bar(x + width, faith_scores, width, label='Faithfulness (RAGAS)', color='#C44E52')

ax.set_ylabel('Значение метрики (от 0 до 1)')
ax.set_title('Сравнительный анализ качества генерации тестов маппараметрическими LLM')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend(loc='lower right')
ax.grid(axis='y', linestyle='--', alpha=0.7)


def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

plt.tight_layout()
plt.savefig('metrics_comparison.pdf')
print("График качества сохранен как metrics_comparison.pdf")


time_df = df.groupby('model_name')['generation_time_sec'].mean().reset_index()
times = time_df['generation_time_sec'].tolist()

fig2, ax2 = plt.subplots(figsize=(8, 5))
rects_t = ax2.bar(models, times, width=0.5, color='#8172B3')

ax2.set_ylabel('Среднее время генерации 1 теста (секунды)')
ax2.set_title('Сравнение вычислительных затрат (Latency)')
ax2.grid(axis='y', linestyle='--', alpha=0.7)

for rect in rects_t:
    height = rect.get_height()
    ax2.annotate(f'{height:.1f} s',
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  textcoords="offset points",
                ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('latency_comparison.pdf')
print("График времени сохранен как latency_comparison.pdf")