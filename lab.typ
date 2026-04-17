// ─────────────────────────────────────────────
//  Лабораторная работа №3 — Основы машинного обучения
//  Исследование методов кластеризации
// ─────────────────────────────────────────────

#set document(title: "ЛР №3 — Исследование методов кластеризации")
#set page(
  paper: "a4",
  margin: (top: 2cm, bottom: 2cm, left: 3cm, right: 1.5cm),
  numbering: "1",
  number-align: center,
)
#set text(font: "Times New Roman", size: 14pt, lang: "ru")
#set par(justify: true, leading: 1em, first-line-indent: 1.25cm)
#show heading: it => {
  set text(size: 14pt, weight: "bold")
  align(center, it)
  v(0.5em)
}

// ══════════════════════════════════════════════
//  ТИТУЛЬНЫЙ ЛИСТ
// ══════════════════════════════════════════════
#page(numbering: none)[
  #align(center)[
    #text(size: 12pt)[
      ФЕДЕРАЛЬНОЕ ГОСУДАРСТВЕННОЕ БЮДЖЕТНОЕ ОБРАЗОВАТЕЛЬНОЕ УЧРЕЖДЕНИЕ \
      ВЫСШЕГО ОБРАЗОВАНИЯ \
      *«УФИМСКИЙ УНИВЕРСИТЕТ НАУКИ И ТЕХНОЛОГИЙ»*
    ]

    #v(0.5cm)
    #text(size: 12pt)[Кафедра ВМиК]

    #v(3cm)

    #text(size: 16pt, weight: "bold")[ОТЧЁТ ПО ЛАБОРАТОРНОЙ РАБОТЕ №3]

    #v(0.5cm)
    #text(size: 14pt)[
      по дисциплине «Основы машинного обучения»
    ]

    #v(0.8cm)
    #text(size: 14pt, weight: "bold")[
      Исследование методов кластеризации
    ]
  ]

  #v(3cm)

  #align(right)[
    #grid(
      columns: (auto, auto),
      gutter: 0.4cm,
      [*Выполнили:*], [студенты],
      [],             [Ахметшин Д.О.],
      [],             [Ярошко Е.В.],
      [],             [Иванов Д.А.],
      [*Проверил:*],  [Миронов Константин Валерьевич],
    )
  ]

  #v(1fr)
  #align(center)[Уфа — 2024]
]

// ══════════════════════════════════════════════
//  СОДЕРЖАНИЕ
// ══════════════════════════════════════════════
#outline(title: "Содержание", indent: 1.5em)

#pagebreak()

= 1. Описание датасета

В качестве набора данных был выбран датасет *Alcohol QCM Sensor* (вариант 12) из репозитория UCI Machine Learning.

Датасет содержит показания пяти различных газовых датчиков QCM для измерения пяти различных видов спиртов: 1-октанол, 1-пропанол, 2-бутанол, 2-пропанол и 1-изобутанол. Целевой переменной (target) является тип спирта, закодированный в формате One-Hot Encoding в исходных файлах.

- *Количество объектов:* 125
- *Количество признаков:* 10 (числовые, вещественные)
- *Количество классов:* 5

В рамках данной лабораторной работы целевая переменная была исключена из признакового пространства и использовалась исключительно для оценки точности кластеризации (по метрике *Adjusted Rand Index*, ARI).

= 2. Предварительная обработка и снижение размерности

Для корректной работы алгоритмов кластеризации и снижения размерности, данные были нормализованы (стандартизированы) с использованием `StandardScaler` (вычитание среднего и деление на стандартное отклонение).

Снижение размерности выполнялось методом главных компонент (PCA). Были сформированы четыре набора данных:
1. Исходные нормализованные данные (10D).
2. Снижение до 2 компонент (PCA 2D).
3. Снижение до 3 компонент (PCA 3D).
4. Оптимальное количество компонент, объясняющих 95% дисперсии (PCA Opt), что на данном датасете составило 2 компоненты (PCA Opt 2D совпал с PCA 2D).

= 3. Исследование алгоритмов кластеризации

В работе исследовались следующие методы кластеризации из библиотеки `scikit-learn` и `hdbscan`:
- *k-Means* (с заданным $k=5$)
- *Agglomerative clustering* (с $k=5$)
- *Mean Shift*
- *DBSCAN*
- *HDBSCAN*
- *OPTICS*
- *Affinity Propagation*
- *Spectral clustering* (с $k=5$)

Для методов, требующих указания числа кластеров, было выбрано $k=5$ (в соответствии с числом реальных классов в датасете). Для методов, основанных на плотности, использовались базовые параметры (с поправками на масштаб данных). Точность распределения объектов по кластерам оценивалась метрикой *Adjusted Rand Index (ARI)*.

= 4. Результаты экспериментов

Ниже представлена сводная таблица результатов (ARI и время выполнения) для каждого алгоритма на разных размерностях.

#align(center)[
  #table(
    columns: (auto, auto, auto, auto, auto),
    align: (left, left, center, center, center),
    [*Датасет*], [*Алгоритм*], [*Найдено кластеров*], [*ARI*], [*Время (с)*],
    "Original (10D)", "k-Means", "5", "0.2547", "0.0358",
    "Original (10D)", "Agglomerative", "5", "0.2236", "0.0021",
    "Original (10D)", "Mean Shift", "3", "0.1389", "0.4369",
    "Original (10D)", "DBSCAN", "3", "0.0099", "0.0025",
    "Original (10D)", "HDBSCAN", "6", "0.2484", "0.0088",
    "Original (10D)", "OPTICS", "20", "0.3357", "0.1998",
    "Original (10D)", "Affinity Propagation", "8", "0.2458", "0.0158",
    "Original (10D)", "Spectral Clustering", "5", "0.1712", "0.0506",

    "PCA 2D", "k-Means", "5", "0.2315", "0.0026",
    "PCA 2D", "Agglomerative", "5", "0.3265", "0.0014",
    "PCA 2D", "Mean Shift", "3", "0.1156", "0.4084",
    "PCA 2D", "DBSCAN", "3", "0.0099", "0.0018",
    "PCA 2D", "HDBSCAN", "6", "0.2279", "0.0034",
    "PCA 2D", "OPTICS", "15", "0.2571", "0.1609",
    "PCA 2D", "Affinity Propagation", "8", "0.3413", "0.0087",
    "PCA 2D", "Spectral Clustering", "5", "0.2139", "0.0066",

    "PCA 3D", "k-Means", "5", "0.2547", "0.0022",
    "PCA 3D", "Agglomerative", "5", "0.2777", "0.0011",
    "PCA 3D", "Mean Shift", "3", "0.1335", "0.5523",
    "PCA 3D", "DBSCAN", "3", "0.0099", "0.0051",
    "PCA 3D", "HDBSCAN", "5", "0.2341", "0.0037",
    "PCA 3D", "OPTICS", "18", "0.2665", "0.1600",
    "PCA 3D", "Affinity Propagation", "8", "0.2962", "0.0075",
    "PCA 3D", "Spectral Clustering", "5", "0.1712", "0.0150",
  )
]

*(Данные для PCA Opt аналогичны результатам PCA 2D, так как 95% дисперсии достигается при 2 компонентах).*

= 5. Выводы

1. *Эффективность методов кластеризации:* Наилучший показатель соответствия исходным меткам (ARI = $0.3413$) показал алгоритм *Affinity Propagation* на размерности PCA 2D. *Agglomerative Clustering* также продемонстрировал высокие результаты на 2D (ARI = $0.3265$). Алгоритмы, основанные на плотности (DBSCAN), без тщательного подбора параметров `eps` и `min_samples` склонны объединять данные в один-два кластера или классифицировать их как шум (ARI около 0).
2. *Влияние размерности:* Снижение размерности с помощью PCA до 2-х или 3-х компонент во многих случаях увеличивало показатель ARI и снижало уровень шума (например, для агломеративной и спектральной кластеризации).
3. *Производительность:* Снижение размерности существенно сократило время выполнения для ресурсоёмких алгоритмов (Affinity Propagation ускорился почти в 2 раза). Самым медленным оказался *Mean Shift*, а самыми быстрыми — иерархическая агломеративная кластеризация и DBSCAN.
4. В целом, данные имеют сложную структуру, о чём свидетельствует то, что алгоритмы, автоматически подбирающие число кластеров (OPTICS, Affinity Propagation), находят от 8 до 20 подгрупп вместо 5-ти целевых классов.

= Приложение А. Исходный код

Ниже представлен листинг написанного скрипта `main.py` для загрузки, предобработки и оценки алгоритмов:

```python
import time
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import (
    KMeans,
    AgglomerativeClustering,
    MeanShift,
    DBSCAN,
    OPTICS,
    AffinityPropagation,
    SpectralClustering
)
import hdbscan
from sklearn.metrics import adjusted_rand_score

def load_data():
    files = list(Path("QCM Sensor Alcohol Dataset").glob("*.csv"))
    if not files:
        raise FileNotFoundError("Dataset files not found.")

    df_list = []
    for f in files:
        df = pd.read_csv(f, sep=';')
        df_list.append(df)

    full_df = pd.concat(df_list, ignore_index=True)
    X = full_df.iloc[:, :10].values
    Y_onehot = full_df.iloc[:, 10:15].values
    y = np.argmax(Y_onehot, axis=1)
    return X, y

def main():
    X_raw, y = load_data()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    pca_2d = PCA(n_components=2)
    X_pca2 = pca_2d.fit_transform(X_scaled)

    pca_3d = PCA(n_components=3)
    X_pca3 = pca_3d.fit_transform(X_scaled)

    pca_opt = PCA(n_components=0.95)
    X_pca_opt = pca_opt.fit_transform(X_scaled)

    datasets = {
        f"Original ({X_scaled.shape[1]}D)": X_scaled,
        "PCA 2D": X_pca2,
        "PCA 3D": X_pca3,
        f"PCA Opt ({X_pca_opt.shape[1]}D)": X_pca_opt
    }

    models = {
        "k-Means (k=5)": KMeans(n_clusters=5, random_state=42, n_init='auto'),
        "Agglomerative (k=5)": AgglomerativeClustering(n_clusters=5),
        "Mean Shift": MeanShift(),
        "DBSCAN": DBSCAN(eps=1.5, min_samples=5),
        "HDBSCAN": hdbscan.HDBSCAN(min_cluster_size=5),
        "OPTICS": OPTICS(min_samples=5),
        "Affinity Propagation": AffinityPropagation(random_state=42),
        "Spectral Clustering (k=5)": SpectralClustering(
            n_clusters=5, random_state=42, assign_labels='discretize'
        )
    }

    results = []

    for data_name, X in datasets.items():
        for model_name, model in models.items():
            start_time = time.time()
            try:
                labels = model.fit_predict(X)
                exec_time = time.time() - start_time

                ari = adjusted_rand_score(y, labels)
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

                results.append({
                    "Dataset": data_name,
                    "Algorithm": model_name,
                    "Clusters Found": n_clusters,
                    "ARI": round(ari, 4),
                    "Time (s)": round(exec_time, 4)
                })
            except Exception as e:
                pass

    results_df = pd.DataFrame(results)
    results_df.to_csv("clustering_results.csv", index=False)

if __name__ == "__main__":
    main()
```
