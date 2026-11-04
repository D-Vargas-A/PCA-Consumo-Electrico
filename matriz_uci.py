import pandas as pd
import numpy as np

print("1. Cargando dataset de UCI (esto puede tardar un par de minutos, es pesado)...")
# El archivo usa ';' como separador y ',' para los decimales
df = pd.read_csv('LD2011_2014.txt', sep=';', decimal=',', parse_dates=[0], index_col=0)

print("2. Procesando datos y simulando carga del campus...")
# Sumar el consumo de todos los clientes en cada franja para simular un campus completo
consumo_total = df.sum(axis=1).rename('consumo')

# Crear DataFrame estructurado
df_campus = consumo_total.reset_index()
df_campus.columns = ['timestamp', 'consumo']

# Extraer fecha, hora y si es fin de semana (sábado=5, domingo=6)
df_campus['fecha'] = df_campus['timestamp'].dt.date
df_campus['hora'] = df_campus['timestamp'].dt.time
df_campus['es_fin_de_semana'] = df_campus['timestamp'].dt.dayofweek.isin([5, 6]).astype(int)

print("3. Construyendo la Matriz X (96 franjas + contexto)...")
# Pivotar para que las filas sean los días y las columnas las 96 franjas horarias
matriz_x = df_campus.pivot(index='fecha', columns='hora', values='consumo')

# Agregar la columna de contexto (es_fin_de_semana)
fines_de_semana = df_campus.groupby('fecha')['es_fin_de_semana'].first()
matriz_x = matriz_x.join(fines_de_semana)

# Eliminar días que no tengan las 96 franjas completas
matriz_x = matriz_x.dropna()

print("\n✅ ¡MATRIZ X CONSTRUIDA CON ÉXITO!")
print(f"Total de días (filas): {matriz_x.shape[0]}")
print(f"Variables (columnas): {matriz_x.shape[1]} (96 franjas + 1 contexto)")
print("\nPrimeras 3 filas de la matriz:")
print(matriz_x.head(3))

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

print("\n--- INICIANDO FASE DE PCA ---")
# SOLUCIÓN: Convertir todos los nombres de las columnas a texto (string)
matriz_x.columns = matriz_x.columns.astype(str)

print("1. Estandarizando la matriz (Media 0, Desviación 1)...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(matriz_x)

print("2. Calculando componentes para retener el 90% de varianza...")
pca_full = PCA().fit(X_scaled)
varianza_acumulada = np.cumsum(pca_full.explained_variance_ratio_)
k = np.argmax(varianza_acumulada >= 0.90) + 1
var_explicada = varianza_acumulada[k-1] * 100
print(f"   -> k = {k} componentes (explican el {var_explicada:.2f}% de la varianza)")

print("3. Ajustando PCA definitivo y calculando proyecciones...")
pca = PCA(n_components=k)
scores = pca.fit_transform(X_scaled) # Representación en el espacio k-dimensional

print("4. Calculando estadísticos T² y SPE...")
# T² de Hotelling: distancia en el hiperplano del PCA
T2 = np.sum((scores ** 2) / pca.explained_variance_, axis=1)

# SPE (Squared Prediction Error): distancia ortogonal al hiperplano
X_reconstruido = pca.inverse_transform(scores)
SPE = np.sum((X_scaled - X_reconstruido) ** 2, axis=1)

print("5. Aplicando umbrales del percentil 99...")
umbral_T2 = np.percentile(T2, 99)
umbral_SPE = np.percentile(SPE, 99)

dias_anomalos_T2 = matriz_x.index[T2 > umbral_T2]
dias_anomalos_SPE = matriz_x.index[SPE > umbral_SPE]
anomalias_combinadas = set(dias_anomalos_T2).union(set(dias_anomalos_SPE))

print(f"   -> Umbral T²: {umbral_T2:.2f} | Días marcados: {len(dias_anomalos_T2)}")
print(f"   -> Umbral SPE: {umbral_SPE:.2f} | Días marcados: {len(dias_anomalos_SPE)}")
print(f"   -> TOTAL días únicos anómalos detectados por PCA: {len(anomalias_combinadas)}")

import os
import matplotlib.pyplot as plt

print("\n--- CALCULANDO LÍNEA BASE Y COMPARANDO ---")
# Obtener nombres de las 96 franjas (ignorando el contexto)
franjas = [c for c in matriz_x.columns if c != 'es_fin_de_semana']

# Calcular medias y desviaciones estándar por tipo de día (laboral vs fin de semana)
medias = matriz_x.groupby('es_fin_de_semana')[franjas].transform('mean')
desviaciones = matriz_x.groupby('es_fin_de_semana')[franjas].transform('std')

# Calcular Z-score absoluto para cada franja
z_scores = np.abs((matriz_x[franjas] - medias) / desviaciones)

# Contar cuántas franjas superan |z| > 3 en un mismo día
franjas_anomalas_por_dia = (z_scores > 3).sum(axis=1)

# Marcar como anomalía si hay al menos 8 franjas (2 horas continuas)
dias_anomalos_base = matriz_x.index[franjas_anomalas_por_dia >= 8]
print(f"-> Días anómalos marcados por la Línea Base: {len(dias_anomalos_base)}")

# Comparación PCA vs Línea Base
coincidencias = set(anomalias_combinadas).intersection(set(dias_anomalos_base))
print(f"-> Coincidencias exactas (PCA y Línea Base): {len(coincidencias)}")

print("\n--- GENERANDO GRÁFICAS PARA LAS DIAPOSITIVAS ---")
os.makedirs('figuras', exist_ok=True)

# 1. Gráfica de Varianza Explicada
plt.figure(figsize=(8,5))
plt.plot(range(1, len(varianza_acumulada)+1), varianza_acumulada * 100, marker='o', color='b')
plt.axhline(y=90, color='r', linestyle='--', label='Umbral 90%')
plt.axvline(x=k, color='g', linestyle='--', label=f'k={k}')
plt.title('Varianza Explicada Acumulada')
plt.xlabel('Número de Componentes')
plt.ylabel('Varianza Explicada (%)')
plt.legend()
plt.grid(True)
plt.savefig('figuras/varianza_explicada.png', dpi=300)
plt.close()

# 2. Gráfica de Estadísticos T2 y SPE
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
ax1.plot(range(len(T2)), T2, label='T²', color='b')
ax1.axhline(umbral_T2, color='r', linestyle='--', label='Umbral (99%)')
ax1.set_title('Estadístico T² de Hotelling')
ax1.set_xlabel('Días del dataset')
ax1.legend()

ax2.plot(range(len(SPE)), SPE, label='SPE', color='orange')
ax2.axhline(umbral_SPE, color='r', linestyle='--', label='Umbral (99%)')
ax2.set_title('Error de Predicción al Cuadrado (SPE)')
ax2.set_xlabel('Días del dataset')
ax2.legend()
plt.tight_layout()
plt.savefig('figuras/estadisticos_t2_spe.png', dpi=300)
plt.close()

# 3. Proyección 2D (Calculamos un PCA temporal de 2 componentes solo para la gráfica)
pca_2d = PCA(n_components=2)
scores_2d = pca_2d.fit_transform(X_scaled)

plt.figure(figsize=(8,6))
plt.scatter(scores_2d[:, 0], scores_2d[:, 1], alpha=0.5, label='Días Normales', color='gray')

# Extraer posiciones de las anomalías para pintarlas de rojo
idx_anomalos = [i for i, d in enumerate(matriz_x.index) if d in anomalias_combinadas]
plt.scatter(scores_2d[idx_anomalos, 0], scores_2d[idx_anomalos, 1], color='red', label='Anomalías (PCA)', marker='x')

plt.title('Proyección 2D del Consumo Eléctrico')
plt.xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}%)')
plt.ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}%)')
plt.legend()
plt.grid(True)
plt.savefig('figuras/proyeccion_2d.png', dpi=300)
plt.close()

print("✅ Gráficas guardadas con éxito en la carpeta 'figuras/'.")

# 1. Número exacto de días (n)
print(f"Número de días (n) usados en el PCA: {matriz_x.shape[0]}")

# 2. Fechas de los 30 días anómalos
print("\nFechas de los 30 días anómalos:")
fechas_anomalas = sorted(list(anomalias_combinadas))
for f in fechas_anomalas:
    print(f)

# 3. Verificar el aumento de consumo por año
print("\nConsumo total por año (Verificando aumento de clientes):")
# Extraemos el año del índice de la matriz_x
matriz_x['año'] = pd.to_datetime(matriz_x.index).year
consumo_por_año = matriz_x[franjas].sum(axis=1).groupby(matriz_x['año']).sum()
print(consumo_por_año)