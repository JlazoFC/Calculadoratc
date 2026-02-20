import streamlit as st
import re
import pandas as pd
import boto3
import io
import requests
import numpy as np
import os
import pytz
import requests
from dotenv import load_dotenv
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import subprocess
import statsmodels.api as sm
import joblib 
import os
import time
import pandas as pd
import numpy as np
import warnings
import subprocess
import tempfile
import re
import pickle
import xgboost as xgb
import pandas as pd
import sys
warnings.filterwarnings('ignore')
import math
import pickle 
import json


AWS_ACCESS_KEY = os.environ['AWS_ACCESS_KEY_ID']
AWS_SECRET_KEY = os.environ['AWS_SECRET_ACCESS_KEY']
BUCKET_NAME = os.environ['S3_BUCKET']
ARCHIVO_S3 = "tconetcacalculadora.csv"
ADV = 'ADVcalculadora.csv'
SBLMID = 'SinBLMDID.csv'
#Hola
# Verificar que las variables se cargaron correctamente
if not all([AWS_ACCESS_KEY, AWS_SECRET_KEY, BUCKET_NAME]):
    print("⚠️ Las variables de entorno no se cargaron correctamente.")
else:
    print("✅ Las variables de entorno se cargaron correctamente.")

# Carga del cliente S3
s3 = boto3.client('s3',
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY
)
# ---------------- CONFIGURACIÓN ----------------

GITHUB_RAW_URL = "https://raw.githubusercontent.com/JlazoFC/Calculadoratc/refs/heads/main/"  # Ajusta esto con tu usuario/repositorio

BASEID = 'Base_martin.csv'
BLMID = 'blmid_validos.csv'
CPID = 'CPID.csv'
AIS = 'https://github.com/JlazoFC/Calculadoratc/blob/main/Base_AIS.parquet'  # Recomendado usar parquet si el archivo es grande
USU = 'Usuarios.csv'

print("✅ Configuración cargada para lectura desde GitHub.")

# ---------------- FUNCIONES ----------------

@st.cache_data
def BimboIDbase():
    url = GITHUB_RAW_URL + BASEID
    try:
        df = pd.read_csv(url, dtype={'blmId': str})
    except Exception:
        df = pd.DataFrame(columns=['blmId', 'Morosidad_Promedio', 'Gradient Boosting_Proba',
                                   'Decil_ventas','PromedioVisitasXMesBimbo','ventaPromedioSemanalUlt12Semanas',
                                   'Giro_de_Cliente','MontoMinCredito','DiasConCreditoVigente'])
    return df

@st.cache_data
def BLMIDVAL():
    url = GITHUB_RAW_URL + BLMID
    try:
        df = pd.read_csv(url, dtype={'blmId': str})
    except Exception:
        df = pd.DataFrame(columns=['blmId'])
    return df



@st.cache_data
def CPbase():
    url = GITHUB_RAW_URL + CPID
    try:
        df = pd.read_csv(url, dtype={'d_codigo': str})
    except Exception:
        df = pd.DataFrame(columns=['d_codigo', 'd_asenta', 'd_tipo_asenta', 'D_mnpio', 'd_estado','d_zona'])
    return df

@st.cache_data
def AISbase():
    # URL corregida con el nombre exacto del archivo
    url = "https://raw.githubusercontent.com/JlazoFC/Calculadoratc/main/Base_AIS.parquet"
    print(f"🔍 Intentando cargar desde: {url}")
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        df = pd.read_parquet(io.BytesIO(response.content))
        df["bimboId"] = df["bimboId"].astype(str).str.strip()
        print(f"✅ Archivo cargado exitosamente. Forma: {df.shape}")
        return df
        
    except Exception as e:
        print(f"⚠️ Error cargando Base_AIS.parquet: {e}")
        return pd.DataFrame(columns=["bimboId"])


@st.cache_data
def USUARIOS():
    url = GITHUB_RAW_URL + USU
    try:
        df = pd.read_csv(url, dtype={'Usuario': str, 'Pass' : str})
    except Exception:
        df = pd.DataFrame(columns=['Usuario', 'Pass'])
    return df

## Se cargara hasta que se tenga el S3 de FC

@st.cache_data(ttl=300) 
def cargar_base():
    try:
        response = s3.get_object(Bucket=BUCKET_NAME, Key=ARCHIVO_S3)
        df = pd.read_csv(io.BytesIO(response['Body'].read()), dtype={'nombre': str, 'BimboID': str,
                                                                     'blmId' : str, 'Solicitud':str,
                                                                     'Usuario_registro':str})
    except s3.exceptions.ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            df = pd.DataFrame(columns=['nombre', 'edad', 'genero', 'BimboID'])
        else:
            raise e
    return df

def guardar_base(df):
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    s3.put_object(Bucket=BUCKET_NAME, Key=ARCHIVO_S3, Body=buffer.getvalue())

def eliminar_registro(Solicitud):
    df = cargar_base()
    df = df[df['Solicitud'] != Solicitud]
    guardar_base(df)

## Cargar base ADV
@st.cache_data(ttl=300)
def cargar_base_ADV():
    try:
        response = s3.get_object(Bucket=BUCKET_NAME, Key=ADV)
        df = pd.read_csv(io.BytesIO(response['Body'].read()), dtype={'BimboID': str,
                                                                     'blmId' : str, 'Solicitud':str,
                                                                     'Usuario_registro':str})
    except s3.exceptions.ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            df = pd.DataFrame(columns=['edad', 'genero', 'BimboID'])
        else:
            raise e
    return df

def guardar_base_ADV(df):
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    s3.put_object(Bucket=BUCKET_NAME, Key=ADV, Body=buffer.getvalue())

def eliminar_registro_ADV(Solicitud):
    df = cargar_base_ADV()
    df = df[df['Solicitud'] != Solicitud]
    guardar_base_ADV(df)

## Cargar base Sin BLMID

@st.cache_data(ttl=300)
def cargar_base_SBLMID():
    try:
        response = s3.get_object(Bucket=BUCKET_NAME, Key=SBLMID)
        df = pd.read_csv(io.BytesIO(response['Body'].read()), dtype={'BimboID': str,
                                                                     'blmId' : str, 'Solicitud':str,
                                                                     'Usuario_registro':str})
    except s3.exceptions.ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            df = pd.DataFrame(columns=['edad', 'genero', 'BimboID'])
        else:
            raise e
    return df

def guardar_base_SBLMID(df):
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    s3.put_object(Bucket=BUCKET_NAME, Key=SBLMID, Body=buffer.getvalue())

def eliminar_registro_SBLMID(Solicitud):
    df = cargar_base_SBLMID()
    df = df[df['Solicitud'] != Solicitud]
    guardar_base_SBLMID(df)

import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

def limpiar_datos_numericos(df):
    """
    Limpia datos numéricos para evitar problemas con XGBoost
    """
    print("🧹 Limpiando datos numéricos...")
    
    # Reemplazar infinitos con NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Para cada columna numérica
    for col in df.columns:
        if df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
            # Verificar valores extremos
            max_val = df[col].max()
            min_val = df[col].min()
            
            # Si hay valores extremadamente grandes, los limitamos
            if abs(max_val) > 1e6:
                percentile_99 = df[col].quantile(0.99)
                df[col] = df[col].clip(upper=percentile_99)
                print(f"⚠️ Limitando valores extremos en '{col}': max = {max_val:.2e} -> {percentile_99:.2f}")
            
            if abs(min_val) > 1e6:
                percentile_1 = df[col].quantile(0.01)
                df[col] = df[col].clip(lower=percentile_1)
                print(f"⚠️ Limitando valores extremos en '{col}': min = {min_val:.2e} -> {percentile_1:.2f}")
    
    return df

import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

def limpiar_datos_numericos(df):
    """
    Limpia datos numéricos para evitar problemas con XGBoost
    """
    print("🧹 Limpiando datos numéricos...")
    
    # Reemplazar infinitos con NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Para cada columna numérica
    for col in df.columns:
        if df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
            # Verificar valores extremos
            max_val = df[col].max()
            min_val = df[col].min()
            
            # Si hay valores extremadamente grandes, los limitamos
            if abs(max_val) > 1e6:
                percentile_99 = df[col].quantile(0.99)
                df[col] = df[col].clip(upper=percentile_99)
                print(f"⚠️ Limitando valores extremos en '{col}': max = {max_val:.2e} -> {percentile_99:.2f}")
            
            if abs(min_val) > 1e6:
                percentile_1 = df[col].quantile(0.01)
                df[col] = df[col].clip(lower=percentile_1)
                print(f"⚠️ Limitando valores extremos en '{col}': min = {min_val:.2e} -> {percentile_1:.2f}")
    
    return df

def preparar_datos_modelo(df_entrada, feature_names):
    """Preparar datos exactamente como el modelo original con validaciones mejoradas"""
    df = df_entrada.copy()
    
    # Si los datos ya tienen todas las features del modelo (datos preprocesados)
    if all(feature in df.columns for feature in feature_names):
        print("✅ Datos ya preprocesados, aplicando limpieza...")
        df_limpio = limpiar_datos_numericos(df[feature_names])
        return df_limpio.fillna(0)
    
    # Para datos nuevos que necesitan preprocesamiento
    print(f"🔧 Preprocesando datos para {len(feature_names)} features...")
    
    # 1. Limpiar datos antes del preprocesamiento
    df = limpiar_datos_numericos(df)
    
    # 2. Variables categóricas que necesitan dummies
    columnas_categoricas = ["PE_TC_PE_Ventas_AG", "PE_TC_PE_MUNICIPIO_AGR2", "PE_TC_PE_ENTIDAD_AGR"]
    columnas_existentes = [col for col in columnas_categoricas if col in df.columns]
    
    if columnas_existentes:
        print(f"📊 Creando dummies para: {columnas_existentes}")
        df = pd.get_dummies(df, columns=columnas_existentes, drop_first=False)
    
    # 3. Asegurar que todas las features del modelo estén presentes
    for feature in feature_names:
        if feature not in df.columns:
            # Si es una variable dummy que no existe, llenar con 0
            if any(cat in feature for cat in ["PE_TC_PE_Ventas_AG", "PE_TC_PE_MUNICIPIO_AGR2", "PE_TC_PE_ENTIDAD_AGR"]):
                df[feature] = 0
            # Si es una variable numérica que no existe, llenar con 0 o valor por defecto
            else:
                df[feature] = 0
                print(f"⚠️ Variable faltante rellenada con 0: {feature}")
    
    # 4. Reordenar columnas según el orden del modelo
    df_final = df[feature_names]
    
    # 5. Limpieza final y validación
    df_final = limpiar_datos_numericos(df_final)
    df_final = df_final.fillna(0)
    
    # 6. Validación final antes de enviar a XGBoost
    print("🔍 Validación final de datos...")
    
    # Verificar infinitos
    inf_check = np.isinf(df_final.values).any()
    if inf_check:
        print("❌ Aún hay valores infinitos después de la limpieza")
        # Reemplazar cualquier infinito restante con 0
        df_final = df_final.replace([np.inf, -np.inf], 0)
    
    # Verificar NaN
    nan_check = df_final.isna().any().any()
    if nan_check:
        print("❌ Aún hay valores NaN después de la limpieza")
        df_final = df_final.fillna(0)
    
    # Verificar valores extremos
    max_val = df_final.values.max()
    min_val = df_final.values.min()
    
    if abs(max_val) > 1e6 or abs(min_val) > 1e6:
        print(f"⚠️ Valores extremos detectados: max={max_val:.2e}, min={min_val:.2e}")
        # Aplicar escalado robusto si es necesario
        df_final = df_final.clip(-1e6, 1e6)
    
    print(f"✅ Datos preparados: {df_final.shape[0]} filas x {df_final.shape[1]} columnas")
    print(f"📊 Rango de valores: [{df_final.values.min():.2f}, {df_final.values.max():.2f}]")
    
    return df_final

# Cargar modelo convertido
ruta_modelo_pkl = "modelo_ais_convertido.pkl"

def ejecutar_modelo_ais_python(df_entrada):
    """
    Ejecuta el modelo AIS usando el archivo .pkl convertido con validaciones mejoradas
    """
    try:
        print(f"📥 Cargando modelo desde: {ruta_modelo_pkl}")
        
        # Cargar modelo
        with open(ruta_modelo_pkl, 'rb') as f:
            modelo_dict = pickle.load(f)
        
        modelo = modelo_dict['modelo']
        feature_names = modelo_dict['feature_names']
        
        print(f"🎯 Modelo cargado con {len(feature_names)} features")
        print(f"📊 Datos de entrada: {df_entrada.shape[0]} filas x {df_entrada.shape[1]} columnas")
        print(f"📋 Columnas de entrada: {list(df_entrada.columns)}")
        
        # Preparar datos con validaciones mejoradas
        df_procesado = preparar_datos_modelo(df_entrada, feature_names)
        
        # Validación extra antes de crear DMatrix
        print("🔧 Creando DMatrix para XGBoost...")
        
        # Verificar que no hay problemas en los datos
        if np.isinf(df_procesado.values).any():
            raise ValueError("Los datos aún contienen valores infinitos")
        
        if np.isnan(df_procesado.values).any():
            raise ValueError("Los datos aún contienen valores NaN")
        
        # Crear DMatrix con manejo de missing values
        dmatrix = xgb.DMatrix(
            df_procesado.values, 
            feature_names=feature_names,
            missing=np.nan  # Especificar cómo manejar valores faltantes
        )
        
        # Predecir
        predicciones = modelo.predict(dmatrix)
        
        print(f"✅ Predicción completada: {predicciones[0]:.4f}")
        
        # Devolver en el mismo formato que la función original
        resultado_df = pd.DataFrame({
            'preds': predicciones
        })
        
        return resultado_df
        
    except Exception as e:
        print(f"❌ Error detallado: {e}")
        print(f"📋 Columnas disponibles en entrada: {list(df_entrada.columns) if hasattr(df_entrada, 'columns') else 'No DataFrame'}")
        
        # Diagnóstico adicional
        if 'df_procesado' in locals():
            print("🔍 Diagnóstico de datos procesados:")
            print(f"   - Forma: {df_procesado.shape}")
            print(f"   - Infinitos: {np.isinf(df_procesado.values).sum()}")
            print(f"   - NaN: {np.isnan(df_procesado.values).sum()}")
            print(f"   - Rango: [{df_procesado.values.min():.2f}, {df_procesado.values.max():.2f}]")
        
        raise Exception(f"Error ejecutando modelo Python: {e}")

# Tabla de reglas
reglas = pd.DataFrame({
    "ScoreFico_min": [0, 0, 549, 549, 649, 0, 606],
    "ScoreFico_max": [548, 548, 648, 648, 1000, 605, 1000],
    "MicroScore_min": [-10, 621, -10, 621, -10, 661, 661],
    "MicroScore_max": [620, 660, 620, 660, 660, 1000, 1000],
    "Grupo": [1, 2, 3, 4, 5, 6, 7]
})

## rango_montos_ADV
def obtener_rango_monto_adv(valor):
    montos = [
        (0,6000, 1),
        (6000.1, 8000, 2),
        (8000.1, 10000, 3),
        (10000.1, 15000, 4),
        (15000.1, 10000000000, 5)
    ]
    
    for minimo, maximo, rango in montos:
        if minimo <= valor <= maximo:
            return rango
    return None  # Si no cae en ningún rango

def obtener_porcentaje(rango: int, decil: int) -> str:
    tabla = {
        1: [79, 76, 74, 72, 69, 68, 67, 62, 60, 58],
        2: [70, 67, 65, 62, 60, 59, 58, 53, 51, 49],
        3: [67, 64, 62, 59, 57, 56, 55, 50, 48, 46],
        4: [63, 60, 58, 55, 53, 52, 51, 46, 44, 42],
        5: [62, 60, 57, 55, 52, 51, 50, 46, 44, 42]
    }

    if rango not in tabla:
        return "Rango inválido"
    if decil < 1 or decil > 10:
        return "Decil inválido"

    return f"{tabla[rango][decil - 1]}%"

### Decil de riesgos
def asignar_decil(score_fico, micro_score):
    for _, regla in reglas.iterrows():
        if (regla["ScoreFico_min"] <= score_fico <= regla["ScoreFico_max"] and
            regla["MicroScore_min"] <= micro_score <= regla["MicroScore_max"]):
            return regla["Grupo"]
    return None  # Si no entra en ningún rango


# ## Decil RL
# def obtener_decil(valor):
#     rangos = [
#         (0, 0.129, 10),
#         (0.13, 0.216, 9),
#         (0.217, 0.267, 8),
#         (0.268, 0.31, 7),
#         (0.311, 0.356, 6),
#         (0.357, 0.405, 5),
#         (0.406, 0.461, 4),
#         (0.462, 0.523, 3),
#         (0.524, 0.599, 2),
#         (0.6, 1, 1),
#     ]
    
#     for minimo, maximo, decil in rangos:
#         if minimo <= valor <= maximo:
#             return decil
#     return None  # Si no cae en ningún rango

## Decil RL
def obtener_decil(valor):
    # Puntos de corte redondeados a 3 decimales
    # Los deciles van del 10 (valores más bajos) al 1 (valores más altos)
    # Rango: (minimo_incluido, maximo_incluido, decil)
    rangos = [
        # Decil 10: hasta 0.112
        (0.000, 0.112, 10),
        # Decil 9: más de 0.112 hasta 0.194
        (0.113, 0.194, 9),
        # Decil 8: más de 0.194 hasta 0.250
        (0.195, 0.250, 8),
        # Decil 7: más de 0.250 hasta 0.292
        (0.251, 0.292, 7),
        # Decil 6: más de 0.292 hasta 0.335
        (0.293, 0.335, 6),
        # Decil 5: más de 0.335 hasta 0.382
        (0.336, 0.382, 5),
        # Decil 4: más de 0.382 hasta 0.437
        (0.383, 0.437, 4),
        # Decil 3: más de 0.437 hasta 0.498
        (0.438, 0.498, 3),
        # Decil 2: más de 0.498 hasta 0.578
        (0.499, 0.578, 2),
        # Decil 1: más de 0.578 (hasta 1.000)
        (0.579, 1.000, 1),
    ]
    
    # Redondear el valor de entrada a 3 decimales para la comparación
    valor_redondeado = round(valor, 3)
    
    for minimo, maximo, decil in rangos:
        # La comparación usa el valor redondeado para asegurar consistencia
        if minimo <= valor_redondeado <= maximo:
            return decil
            
    # Asume que el valor está entre 0 y 1
    return None # Si no cae en ningún rango, es un valor fuera de [0, 1] o un error.

# ## Decil AIS
# def obtener_decil_AIS(valor):
#     rangos = [
#         (0, 0.315, 10),
#         (0.316, 0.359, 9),
#         (0.360, 0.390403807, 8),
#         (0.390479475, 0.390, 7),
#         (0.391, 0.422, 6),
#         (0.423, 0.456, 5),
#         (0.457, 0.503, 4),
#         (0.546, 0.583, 3),
#         (0.584, 0.627, 2),
#         (0.628, 1, 1),
#     ]
    
#     for minimo, maximo, decil in rangos:
#         if minimo <= valor <= maximo:
#             return decil
#     return None  # Si no cae en ningún rango

## Decil AIS
def obtener_decil_AIS(valor):
    # Puntos de corte redondeados a 3 decimales
    # Los deciles van del 10 (valores más bajos) al 1 (valores más altos)
    # Rango: (minimo_incluido, maximo_incluido, decil)
    rangos = [
        # Decil 10: hasta 0.315
        (0.000, 0.315, 10),
        # Decil 9: más de 0.315 hasta 0.359
        (0.316, 0.359, 9),
        # Decil 8: más de 0.359 hasta 0.390
        (0.360, 0.390, 8),
        # Decil 7: más de 0.390 hasta 0.422
        (0.391, 0.422, 7),
        # Decil 6: más de 0.422 hasta 0.456
        (0.423, 0.456, 6),
        # Decil 5: más de 0.456 hasta 0.503
        (0.457, 0.503, 5),
        # Decil 4: más de 0.503 hasta 0.545
        (0.504, 0.545, 4),
        # Decil 3: más de 0.545 hasta 0.583
        (0.546, 0.583, 3),
        # Decil 2: más de 0.583 hasta 0.627
        (0.584, 0.627, 2),
        # Decil 1: más de 0.627 (hasta 1.000)
        (0.628, 1.000, 1),
    ]
    
    # Redondear el valor de entrada a 3 decimales para la comparación,
    # asegurando que los límites contiguos funcionen sin gaps.
    valor_redondeado = round(valor, 3)
    
    for minimo, maximo, decil in rangos:
        if minimo <= valor_redondeado <= maximo:
            return decil
            
    return None # Si no cae en ningún rango, asumiendo que está fuera de [0, 1].

def asignar_desiscion(grupo, Microscore):
    if grupo == 6 or (-7 <= Microscore <= -1):
        return "Rechazado"
    else:
        return "Aceptado"
    
def asignar_desiscion_ADV(Decil):
    if Decil == 1 :
        return "Rechazado"
    else:
        return "Aceptado"
    
def asignar_desiscion_SBLMID(Decil):
    if Decil < 3 :
        return "Rechazado"
    else:
        return "Aceptado"

############################## FUNCIONES PARA MODELO RL ###############################
# Luego, define la clase MultiColumnLabelEncoder
class MultiColumnLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, handle_unknown='error'):
        self.encoders = None
        self.handle_unknown = handle_unknown
    
    def fit(self, X, y=None):
        self.encoders = [LabelEncoder() for _ in range(X.shape[1])]
        for i, encoder in enumerate(self.encoders):
            encoder.fit(X[:, i])
        return self
    
    def transform(self, X):
        X_encoded = np.zeros_like(X, dtype=int)
        for i, encoder in enumerate(self.encoders):
            try:
                X_encoded[:, i] = encoder.transform(X[:, i])
            except ValueError:
                if self.handle_unknown == 'ignore':
                    X_encoded[:, i] = -1  # Valor para categorías desconocidas
                else:
                    raise
        return X_encoded
    
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

# Después, define las transformaciones
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer_label = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
    ('label', MultiColumnLabelEncoder(handle_unknown='ignore'))
])
 
# Función para preprocesar nuevos datos
def preprocesar_nuevos_datos(datos_nuevos, modelo_cargado):
    """
    Preprocesa nuevos datos utilizando el preprocesador guardado
    """
    # Aplicar el mismo preprocesamiento que se utilizó en el entrenamiento
    preprocessor = modelo_cargado['preprocessor'] # procesador de datos (se requieren las funciones de pipe line de cada tipo de dato)
    X_prepared = preprocessor.transform(datos_nuevos)
   
    # Convertir a DataFrame con los nombres de las características
    X_df = pd.DataFrame(X_prepared, columns=modelo_cargado['feature_names'])
   
    # Seleccionar solo las características que se utilizaron en el modelo final
    X_selected = X_df[modelo_cargado['selected_features']]
   
    # Eliminar columnas altamente correlacionadas
    X_selected = X_selected.drop(columns=modelo_cargado['drop_columns'], errors='ignore')
   
    return X_selected
 
# Función para predecir probabilidades
def predecir_probabilidades(datos_preprocesados, modelo_cargado):
    """
    Predice las probabilidades utilizando el modelo cargado
    """
    # Añadir constante para el intercepto
    X_const = sm.add_constant(datos_preprocesados)
   
    # Predecir probabilidades
    probabilidades = modelo_cargado['model'].predict(X_const)
   
    return probabilidades

# Finalmente, define la función para cargar el modelo
def cargar_modelo(ruta_modelo='modelo_regresion_logistica_v2.pkl'):
    """
    Carga el modelo de regresión logística guardado previamente
    """
    try:
        with open(ruta_modelo, 'rb') as file:
            modelo_cargado = pickle.load(file)
        return modelo_cargado
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo {ruta_modelo}")
        return None
    except Exception as e:
        print(f"Error al cargar el modelo: {str(e)}")
        return None

# Y ahora sí, carga el modelo
modelo_cargado = cargar_modelo(ruta_modelo='modelo_regresion_logistica_v2.pkl')


############################## FUNCIONES PARA ASIGNACIÓN FINAL DE DECIL ###############################
# Inicializar la matriz de clusters con -1 (sin asignar)
matriz_clusters = np.full((10, 10), -1, dtype=int)

# Grupo 1
grupo_1_coords = [
    (0, 0), (0, 1), (0, 2),  # RL-1 con XGB-1,2,3
    (1, 0), (1, 1), (1, 2),   # RL-2 con XGB-1,2,3
    (2, 0), (2, 1), (2, 2)
]

# Grupo 2
grupo_2_coords = [
(0,3),	(0,4),
(1,3),	(1,4),
(2,3),	(2,4),
(3,0),	(3,1),	
(3,2),   (3,3),
(3,4),   (4,0),
(4,1),	(4,2),
(4,3),	(4,4)
]

# Grupo 3
grupo_3_coords = [
(0,5),	(0,6),
(1,5),	(1,6),
(2,5),	(2,6),
(3,5),	(3,6),
(4,5),	(4,6),
(5,0),	(5,1),	
(5,2),	(5,3),	
(5,4),  (5,5),
(5,6),   (6,0),	
(6,1),	(6,2)
]

# Grupo 4
grupo_4_coords = [
    (3,7),	(3,8),	(3,9),
    (4,7),	(4,8),	(4,9),
    (5,7),	(5,8),	(5,9)
]

# Grupo 5
grupo_5_coords = [
    (6,3),	(6,4),	(6,5),	
    (6,6),	(6,7),	(6,8),
    (7,3),	(7,4),	(7,5),	
    (7,6),  (8,3)
    ]

# Grupo 6
grupo_6_coords = [
    (6,9),
    (7,7),	(7,8),	(7,9),
    (8,4),	(8,5),	(8,6),	(8,7),	(8,8),	(8,9),
    (9,3),	(9,4),	(9,5),	(9,6),	(9,7),	(9,8),	(9,9)
]

# Grupo 7
grupo_7_coords = [
    (7,0),	(7,1),	(7,2),
    (8,0),	(8,1),	(8,2),
    (9,0),	(9,1),	(9,2)
]

# Grupo 8
grupo_8_coords = [
    (0,7),	(0,8),	(0,9),
    (1,7),	(1,8),	(1,9),
    (2,7),	(2,8),	(2,9)
]


# Asignar grupos a la matriz
for i, j in grupo_1_coords:
    matriz_clusters[i, j] = 0  # Grupo 1 (índice 0)

for i, j in grupo_2_coords:
    matriz_clusters[i, j] = 1  # Grupo 2 (índice 1)

for i, j in grupo_3_coords:
    matriz_clusters[i, j] = 2  # Grupo 3 (índice 2)

for i, j in grupo_4_coords:
    matriz_clusters[i, j] = 3  # Grupo 4 (índice 3)

for i, j in grupo_5_coords:
    matriz_clusters[i, j] = 4  # Grupo 5 (índice 4)

for i, j in grupo_6_coords:
    matriz_clusters[i, j] = 5  # Grupo 6 (índice 5)

for i, j in grupo_7_coords:
    matriz_clusters[i, j] = 6  # Grupo 7 (índice 6)

for i, j in grupo_8_coords:
    matriz_clusters[i, j] = 7  # Grupo 8 (índice 7)

# 11. Función para calificar un solo crédito
deciles_rl = [0.111872366035592, 0.1936463963737612, 0.2503706582687602, 0.2917784224934482, 0.3346987520198585, 0.3817219439752034, 0.4370613082398364, 0.4983681906481146, 0.578388701340094]
deciles_xgb = [0.3153418481349942, 0.3586630284786228, 0.39042650759220165, 0.4220331549644468, 0.455979526042938, 0.503093862533569, 0.5446103394031524, 0.5826071023941041, 0.626696944236755]
def calificar_credito(prob_rl, prob_xgb, deciles_rl, deciles_xgb):
    """Califica un solo crédito según su probabilidad RL y XGB."""
    # Asignar decil RL
    decil_rl = 1  # Valor predeterminado
    for i, umbral in enumerate(deciles_rl):
        if prob_rl >= umbral:
            decil_rl = i + 2
    
    # Asignar decil XGB
    decil_xgb = 1  # Valor predeterminado
    for i, umbral in enumerate(deciles_xgb):
        if prob_xgb >= umbral:
            decil_xgb = i + 2
    
    # Obtener grupo
    grupo_num = -1  # Sin asignar por defecto
    if 0 <= decil_rl-1 < 10 and 0 <= decil_xgb-1 < 10:
        grupo_num = int(matriz_clusters[decil_rl-1, decil_xgb-1])
    
    # Mapeo de números de grupo a nombres
    nombres_grupos = {
        0: 'Grupo 1 - Riesgo Muy Bajo',
        1: 'Grupo 2 - Riesgo Bajo',
        2: 'Grupo 3 - Riesgo Medio-Bajo',
        3: 'Grupo 4 - Riesgo Medio',
        4: 'Grupo 5 - Riesgo Medio-Alto',
        5: 'Grupo 6 - Riesgo Alto',
        6: 'Grupo 7 - Alta Discrepancia XGB',
        7: 'Grupo 8 - Alta Discrepancia RL',
    }
    
    if grupo_num >= 0:
        nombre_grupo = nombres_grupos.get(grupo_num, f'Grupo {grupo_num+1}')
    else:
        nombre_grupo = 'Sin Grupo Asignado'

    grupo_num2 = grupo_num + 1
    
    return nombre_grupo, grupo_num2

def montos_grupo(numerodegrupo, Desiscion):
    rangos = [
        (1, 37500, 50300),
        (2, 28100, 37500),
        (3, 19600, 28100),
        (4, 13900, 19600),
        (5, 10500, 13900),
        (6, 4000, 10500),
        (7, 23700, 46000),
        (8, 10900, 24000)
    ]

    if Desiscion == "Aceptado":
        for grupo, minimo, maximo in rangos:
            if numerodegrupo == grupo:
                return math.ceil(minimo), math.ceil(maximo)
    return 0, 0  # Si no se encuentra el grupo o la decisión no es "Aceptado"

def oferta_final(min_oferta, max_oferta, oferta_original):
    if min_oferta <= oferta_original <= max_oferta:
        return oferta_original
    elif oferta_original < min_oferta:
        return min_oferta
    elif max_oferta < oferta_original:
        return max_oferta

################################# Manejo de la limpieza del formulario
if 'limpiar_formulario' in st.session_state and st.session_state['limpiar_formulario']:
    # Limpieza para TConecta
    st.session_state['Solicitud'] = 4000
    st.session_state['edad'] = 18
    st.session_state['Oferta'] = 0
    st.session_state['comentarios'] = ""
    st.session_state['LLAMADA'] = "No"
    st.session_state['CP'] = ""
    st.session_state['genero'] = "Masculino"
    st.session_state['Dependientes'] = "0"
    st.session_state['Edo_civil'] = "Casado"
    st.session_state['Tipo_negocio'] = "ABARROTES"
    st.session_state['tipo_negocio_especificado'] = ""
    st.session_state['BimboID'] = ""
    st.session_state['blmId'] = ""
    st.session_state['InfoCre'] = "Si"
    st.session_state['MicroScore'] = 0
    st.session_state['ScoreFico'] = 0
    st.session_state["INE"] = False
    st.session_state["Domicilio"] = False
    st.session_state["CURP"] = False
    st.session_state["RHC"] = False
    st.session_state["SPEI"] = False
    st.session_state["FOTO"] = "No"

    # Limpieza para ADV
    st.session_state['Solicitud_ADV'] = 0
    st.session_state['edad_ADV'] = 18
    st.session_state['Oferta_ADV'] = 0
    st.session_state['comentarios_ADV'] = ""
    st.session_state['LLAMADA_ADV'] = "No"
    st.session_state['CP_ADV'] = ""
    st.session_state['genero_ADV'] = "Masculino"
    st.session_state['Dependientes_ADV'] = "0"
    st.session_state['Edo_civil_ADV'] = "Casado"
    st.session_state['Tipo_negocio_ADV'] = "ABARROTES"
    st.session_state['tipo_negocio_especificado_ADV'] = ""
    st.session_state['BimboID_ADV'] = ""
    st.session_state['blmId_ADV'] = ""
    st.session_state['InfoCre_ADV'] = "Si"
    st.session_state['MicroScore_ADV'] = 0
    st.session_state['ScoreFico_ADV'] = 0
    st.session_state["INE_ADV"] = False
    st.session_state["Domicilio_ADV"] = False
    st.session_state["CURP_ADV"] = False
    st.session_state["SPEI_ADV"] = False
    st.session_state["FOTO_ADV"] = "No"

    # Limpieza para BLMID
    st.session_state['Solicitud_BLMID'] = 0
    st.session_state['edad_BLMID'] = 18
    st.session_state['Oferta_BLMID'] = 0
    st.session_state['comentarios_BLMID'] = ""
    st.session_state['LLAMADA_BLMID'] = "No"
    st.session_state['CP_BLMID'] = ""
    st.session_state['genero_BLMID'] = "Masculino"
    st.session_state['Dependientes_BLMID'] = "0"
    st.session_state['Edo_civil_BLMID'] = "Casado"
    st.session_state['Tipo_negocio_BLMID'] = "ABARROTES"
    st.session_state['tipo_negocio_especificado_BLMID'] = ""
    st.session_state['BimboID_BLMID'] = ""
    st.session_state['blmId_BLMID'] = ""
    st.session_state['InfoCre_BLMID'] = "Si"
    st.session_state['MicroScore_BLMID'] = 0
    st.session_state['ScoreFico_BLMID'] = 0
    st.session_state["INE_BLMID"] = False
    st.session_state["Domicilio_BLMID"] = False
    st.session_state["CURP_BLMID"] = False
    st.session_state["SPEI_BLMID"] = False
    st.session_state["FOTO_BLMID"] = "No"

    # Quitar bandera
    st.session_state['limpiar_formulario'] = False

# ---------------- INTERFAZ ----------------
# Configuración general
st.set_page_config(page_title="App de Clientes", layout="centered")

st.markdown("""
    <style>
    .stButton>button {
        background-color: white;
        color: black;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5em 1em;
    }
            
    /* Cambiar el fondo del contenedor principal del formulario */
    div[data-testid="stForm"] {
        background-color: #002b54; /* Color de fondo */
        padding: 20px;              /* Padding alrededor del formulario */
        border-radius: 10px;        /* Esquinas redondeadas */
    }

    h1, h2, h3, h4, h5, h6, h7 {
        color: white !important;
    }
            
    /* Personalizar los campos del formulario */
    .stTextInput, .stNumberInput, .stDateInput, .stSelectbox, .stCheckbox{
        background-color: #0056b1; /* Fondo blanco para los inputs */
        border: 2px solid #ffffff; /* Borde gris */
        border-radius: 8px;        /* Esquinas redondeadas */
        padding: 10px;              /* Padding interno */
        padding-bottom: 20px;
    }
            
    /* Personalizar los campos del formulario */
    .stCheckbox{
        background-color: #47a1ff; /* Fondo blanco para los inputs */
        border: 2px solid #ffffff; /* Borde gris */
        border-radius: 8px;        /* Esquinas redondeadas */
        padding: 10px;              /* Padding interno */
        padding-bottom: 20px;
    }
            
    /* Cambiar color de los labels de los campos a blanco */
    label, .stCheckbox > div {
        color: white !important;
        font-weight: bold;
    }

    .stMarkdown p {
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Obtener la hora actual en UTC y convertirla
fecha_local = pd.Timestamp.now(tz='America/Mexico_City')

# Estado inicial
if "pestana_activa" not in st.session_state:
    st.session_state.pestana_activa = "inicio"

if "autenticado" not in st.session_state:
    st.session_state['autenticado'] = False

if "usuario_actual" not in st.session_state:
    st.session_state['usuario_actual'] = ""

# Funciones navegación
def ir_a_pestana1():
    st.session_state.pestana_activa = "Calculadora TConecta"

def ir_a_pestana2():
    st.session_state.pestana_activa = "Calculadora ADV"

def ir_a_pestana3():
    st.session_state.pestana_activa = "Calculadora SIN BLMID"

def volver_inicio():
    st.session_state.pestana_activa = "inicio"

# Encabezado
col1, col2 = st.columns([5, 3])
with col1:
    st.title("Calculadoras")
with col2:
    st.image("imagen.png", width=200)

# Login
if not st.session_state['autenticado']:
    st.title("🔒 Inicio de Sesión")
    with st.form("login_form"):
        usuario = st.text_input("Usuario")
        contraseña = st.text_input("Contraseña", type="password")
        submit_login = st.form_submit_button("Iniciar Sesión")

        if submit_login:
            usuarios_df = USUARIOS()
            if not usuarios_df.empty and ((usuarios_df['Usuario'] == usuario) & (usuarios_df['Pass'] == contraseña)).any():
                st.session_state['autenticado'] = True
                st.session_state['usuario_actual'] = usuario
                st.success("✅ Inicio de sesión exitoso!")
                st.rerun()
            else:
                st.error("❌ Usuario o contraseña incorrectos.")

# Usuario autenticado
else:
    st.title(f"📋 Bienvenido {st.session_state['usuario_actual']}")

    # Selección de pestaña
    if st.session_state.pestana_activa == "inicio":

        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Calculadora TConecta"):
                ir_a_pestana1()
                st.rerun()
        with col2:
            if st.button("Calculadora ADV"):
                ir_a_pestana2()
                st.rerun()

        with col3:
            if st.button("Calculadora SIN BIMBOID"):
                ir_a_pestana3()
                st.rerun()

    # Calculadora TConecta
    elif st.session_state.pestana_activa == "Calculadora TConecta":
        st.header("🔹 TConecta")
        with st.form("form_cliente"):
            col1, col2 = st.columns(2)
            with col1:
                Solicitud = st.number_input("Solicitud", min_value=4000, max_value=10000000000000, key="Solicitud", step=1000)

                nombre = st.text_input("Nombre completo", max_chars=50, key = "nombre",value="")
                if nombre and len(nombre) < 3:
                    st.warning("El nombre debe tener al menos 3 caracteres.")

                Edad = st.number_input("Edad", min_value=18, max_value=100, step=1,key = "edad")

                InfoCre = st.selectbox("¿Cuenta con información de Crédito?", ["Si", "No"], key="InfoCre")

                MicroScore = st.number_input(
                    "MicroScore",
                    min_value=-9,
                    max_value=1000,
                    step=1,
                    key="MicroScore"
                )

                ScoreFico = st.number_input("ScoreFico", 
                    min_value=0, 
                    max_value=1000, 
                    step=1, 
                    key="ScoreFico"
                )

                Edo_civil = st.selectbox("Estado civil", ["Casado", "Union libre", "Soltero", "Viudo","Divorciado"],key = "Edo_civil")


                Oferta = st.number_input("Oferta", min_value=0, max_value=10000000, step=100,key = "Oferta")

                Comentarios = st.text_input("Comentario", max_chars=150, key = "comentarios",value="")

                llamada = st.selectbox("¿Se realizo llamada?", ["Si", "No"],key = "llamada")

            with col2:
                genero = st.selectbox("Género", ["Masculino", "Femenino", "Otro"],key = "genero")

                Dependientes = st.selectbox("Dependiente", ["0", "1", "2", "3", "4", "5", "+5"],key = "Dependientes")

                CP = st.text_input("Código postal", max_chars=50,key = "CP",value="")
                resultadosCP = pd.DataFrame()

                if CP:
                    if len(CP) != 5:
                        st.warning("El CP debe tener al menos 5 caracteres.")
                    else:
                        tabla_CPID = CPbase()
                        tabla_CPID["d_codigo"] = tabla_CPID["d_codigo"].astype(str)
                        resultadosCP = tabla_CPID[tabla_CPID["d_codigo"] == CP]
                        if not resultadosCP.empty:
                            st.success("✅ Datos encontrados en BaseCP")
                        else:
                            st.error("No se encontraron datos para ese CP.")

                Tipo_negocio = st.selectbox("Tipo de negocio", 
                                ['ABARROTES','CONSUMOS','MISCELANEAS','OTROS','SIN INFORMACION'],
                                key="Tipo_negocio",
                                on_change=None)  # Asegura que se actualice la interfaz

                if st.session_state.Tipo_negocio == "SIN INFORMACION":
                    tipo_negocio_especificado = st.text_input("Especifique el tipo de negocio:", 
                                                            key="tipo_negocio_especificado")
                
                BimboID = st.text_input("BimboID", max_chars=30,key = "BimboID",value="")

                if BimboID:
                    if len(BimboID) < 3:
                        st.warning("El BimboID debe tener al menos 3 caracteres.")
                    else:
                        tabla_AIS = AISbase()
                        tabla_AIS["bimboId"] = tabla_AIS["bimboId"].astype(str)
                        resultadosAIS = tabla_AIS[tabla_AIS["bimboId"] == BimboID]
                        if not resultadosAIS.empty:
                            st.success("✅ Datos encontrados en AIS")
                        else:
                            st.error("No se encontraron datos para ese BimboID.")

                blmId = st.text_input("blmId", max_chars=15,key = "blmId",value="")
                resultados = pd.DataFrame()

                if blmId:
                    if len(blmId) < 3:
                        st.warning("El blmId debe tener al menos 3 caracteres.")
                    else:
                        tabla_bimbo = BimboIDbase()
                        tabla_bimbo["blmId"] = tabla_bimbo["blmId"].astype(str)
                        resultados = tabla_bimbo[tabla_bimbo["blmId"] == blmId]
                        if not resultados.empty:
                            st.success("✅ Datos encontrados en Base Bimbo")
                        else:
                            st.error("No se encontraron datos para ese blmId.")

                valido_ine = st.checkbox("Validación de INE" , key="INE")
                valido_domicilio = st.checkbox("Validación de Domicilio", key="Domicilio")
                valido_curp = st.checkbox("Validación de CURP", key="CURP")
                valido_spei = st.checkbox("Validación de SPEI", key="SPEI")
                Hist_cred = st.checkbox("Revisión de reporte de historial crediticio", key="RHC")
                valido_foto = st.selectbox("Oferta corresponde al tamaño del negocio", ["Si", "No", "No se aprecía"],key = "FOTO")

                # Validaciones de campos obligatorios
                campos_validos = (
                    nombre and len(nombre) >= 3 and
                    Edad and
                    CP and len(CP) == 5 and not resultadosCP.empty and
                    genero and
                    Oferta and
                    Dependientes and
                    Hist_cred and
                    Edo_civil and
                    (Tipo_negocio != "SIN INFORMACION" or (Tipo_negocio == "SIN INFORMACION" and 'tipo_negocio_especificado' in locals() and tipo_negocio_especificado)) and
                    BimboID and
                    blmId and len(blmId) >= 3 and not resultados.empty
                )
                

            submit_button = st.form_submit_button("Guardar registro")

            df1 = cargar_base()
            
            if submit_button:
                if not campos_validos:
                    st.warning("⚠️ Por favor, completa todos los campos correctamente antes de enviar.")
                elif str(Solicitud) in df1['Solicitud'].values:
                    st.warning("⚠️ Esta solicitud ya existe. Por favor, ingresa un número único.")
                else:
                    try:
                        df = cargar_base()
                        
                        # Obtener Score y Casa si se encontró el BimboID
                        Morosidad_Promedio = resultados.iloc[0]["Morosidad_Promedio"] if not resultados.empty else None
                        Gradient_Boosting_Proba = resultados.iloc[0]["Gradient Boosting_Proba"] if not resultados.empty else None
                        Decil_ventas = resultados.iloc[0]["Decil_ventas"] if not resultados.empty else None
                        PromedioVisitasXMesBimbo = resultados.iloc[0]["PromedioVisitasXMesBimbo"] if not resultados.empty else None
                        ventaPromedioSemanalUlt12Semanas = resultados.iloc[0]["ventaPromedioSemanalUlt12Semanas"] if not resultados.empty else None
                        Giro_de_Cliente = resultados.iloc[0]["Giro_de_Cliente"] if not resultados.empty else None
                        MontoMinCredito = resultados.iloc[0]["MontoMinCredito"] if not resultados.empty else None
                        DiasConCreditoVigente = resultados.iloc[0]["DiasConCreditoVigente"] if not resultados.empty else None
                        Estado = resultadosCP.iloc[0]["d_estado"] if not resultadosCP.empty else None
                        Municipio = resultadosCP.iloc[0]["D_mnpio"] if not resultadosCP.empty else None
                        Ingreso_empleado = resultadosCP.iloc[0]["Ingreso_empleado"] if not resultadosCP.empty else None
                        Morosidad = resultadosCP.iloc[0]["Morosidad"] if not resultadosCP.empty else None

                        nuevo = {
                            'Usuario_registro': str(st.session_state['usuario_actual']),
                            'Solicitud':int(Solicitud),
                            'nombre': nombre,
                            'genero': genero,
                            'Oferta': Oferta,
                            'Informacion de Credito': InfoCre,
                            'MicroScore' : MicroScore,
                            'ScoreFico' : ScoreFico,
                            'Dependientes' : Dependientes,
                            'Estado civil' : Edo_civil,
                            'Tipo de negocio' : tipo_negocio_especificado if (Tipo_negocio == "SIN INFORMACION" and 'tipo_negocio_especificado' in locals() and tipo_negocio_especificado) else Tipo_negocio,
                            'BimboID': str(BimboID),
                            'blmId' : str(blmId),
                            'Estado de Residencia': Estado,
                            'Municipio de Residencia': Municipio,
                            'Morosidad' : Morosidad,
                            'Morosidad_Promedio': Morosidad_Promedio,
                            'Ingreso_corriente' : Ingreso_empleado,
                            'Edad': Edad,
                            'Gradient Boosting_Proba': Gradient_Boosting_Proba,
                            'Decil_riesgos': asignar_decil(ScoreFico, MicroScore),
                            'PromedioVisitasXMesBimbo': PromedioVisitasXMesBimbo,
                            'Decil_ventas': Decil_ventas,
                            'DiasConCreditoVigente': DiasConCreditoVigente,
                            'ventaPromedioSemanalUlt12Semanas': ventaPromedioSemanalUlt12Semanas,
                            'MontoMinCredito': MontoMinCredito,
                            'Giro_de_Cliente': Giro_de_Cliente,
                            'Validacion_INE' : valido_ine,
                            'Validacion_domicilio' : valido_domicilio,
                            'Validacion_curp' : valido_curp,
                            'Validacion_spei' : valido_spei,
                            'Validacion_foto' : valido_foto,
                            'Comentarios' : Comentarios,
                            'Llamada' : llamada,
                            'Fecha_registro': fecha_local
                        }

                        datos_nuevos = pd.DataFrame([nuevo])

                        # Preprocesar los datos para calificarlos con el modelo de regresion
                        datos_preprocesados = preprocesar_nuevos_datos(datos_nuevos, modelo_cargado)
                        
                        datos_preprocesados['const'] = 1  # Añadir una constante para el intercepto

                        try:
                            probabilidades = predecir_probabilidades(datos_preprocesados, modelo_cargado)
                            st.success("✅ Modelo Interno ejecutado correctamente.")
                        except Exception as e:
                            st.warning("⚠️ No se pudo ejecutar el modelo. Se usará un valor por defecto del 55%.")
                            probabilidades = [0.35591865]  # Valor por defecto

                        Decil = obtener_decil(probabilidades[0])

                        datos_nuevos["Probabilidad"] = probabilidades
                        datos_nuevos["Decil_modelo"] = Decil

                        resultadosAIS["EDAD"] = Edad 

                        with st.spinner("Ejecutando modelo AIS en Python..."):
                            try:
                                resultado = ejecutar_modelo_ais_python(
                                    df_entrada=resultadosAIS
                                )
                                st.success(f"✅ Modelo AIS ejecutado correctamente.")

                            except Exception as e:
                                st.error(f"❌ Error inesperado durante la ejecución del modelo AIS: {e}")
                                # Usar pandas DataFrame para mantener consistencia
                                resultado = pd.DataFrame({'preds': [0.57]})
                                st.warning("⚠️ Se utilizará el valor por defecto: 0.57")

                        print("Se corrió bien modelo AIS")
                        a = float(resultado['preds'].iloc[0])

                        Decil_AIS = obtener_decil_AIS(a)
                        print("Se corrio bien decilAIS")
                        datos_nuevos["Probabilidad_AIS"] = a
                        datos_nuevos["Decil_AIS"] = Decil_AIS

                        grupo_nombre, grupo_num = calificar_credito(prob_rl=float(probabilidades.iloc[0]), prob_xgb=a, deciles_rl = deciles_rl, deciles_xgb = deciles_xgb)
                        
                        # Agregar estas líneas de debug
                        print(f"Debug - grupo_nombre: {grupo_nombre}")
                        print(f"Debug - grupo_num: {grupo_num}")

                        datos_nuevos["Grupo_nombre"] = grupo_nombre
                        datos_nuevos["Grupo_numero"] = grupo_num

                        Desiscion = asignar_desiscion(grupo_num, MicroScore)

                        datos_nuevos['Desiscion'] = Desiscion

                        print(Desiscion)

                        Min_oferta, Max_oferta = montos_grupo(grupo_num, Desiscion)

                        datos_nuevos['Oferta_min'] = Min_oferta

                        datos_nuevos['Oferta_max'] = Max_oferta

                        Oferta_real = oferta_final(Min_oferta, Max_oferta, Oferta)

                        print("Se asigno la funcion de oferta_final")

                        datos_nuevos['Oferta_final'] = Oferta_real

                        print("Se asigno bien desicion y la oferta real")
                    
                        df = pd.concat([df, datos_nuevos], ignore_index=True)
                        guardar_base(df)
                        st.success("✅ Registro guardado correctamente en AWS_S3.")

                        # Guardamos la info en el estado de sesión para mostrar después
                        st.session_state['mostrar_resultado'] = True
                        st.session_state['solicitud_guardada'] = int(Solicitud)
                        st.session_state['nombre_guardado'] = nombre
                        st.session_state['blmId_guardado'] = str(blmId)
                        st.session_state['Desicion_guardada'] = str(Desiscion)
                        st.session_state['Oferta_input'] = str(Oferta)
                        st.session_state['Oferta_final'] = Oferta_real
                        st.session_state['grupo_nombre_guardado'] = grupo_nombre
                        st.session_state['probamodelo1_guardado'] = float(probabilidades[0]*100)
                        st.session_state['probamodelo2_guardado'] = a*100
                        
                        # Agregar bandera para indicar que se debe limpiar el formulario
                        st.session_state['limpiar_formulario'] = True
                        
                        # Recargar la página para mostrar los campos limpios
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error al guardar el registro: {e}")

        # Mostrar el contenedor con el resultado si existe
        if 'mostrar_resultado' in st.session_state and st.session_state['mostrar_resultado']:
            with st.container():
                st.markdown("### Resultado de la Solicitud")
                
                # Mostrar resultado con fondo transparente y letra blanca usando HTML
                st.markdown(
                    f"""
                    <div style="color: white;">
                        <b>Solicitud:</b> {st.session_state.get('solicitud_guardada', 0)}<br>
                        <b>Grupo de riesgo:</b> {st.session_state.get('grupo_nombre_guardado', 'N/A')}<br>
                        <b>blmId:</b> {st.session_state.get('blmId_guardado', 'N/A')}<br>
                        <b>Oferta:</b> ${int(st.session_state.get('Oferta_input', '0')):,.0f}<br>
                        <b>Oferta sugerida:</b> ${st.session_state.get('Oferta_final', 0):,.0f}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # Mostrar interpretación visual de la probabilidad
                Desicion_value = st.session_state['Desicion_guardada']
                if Desicion_value == "Aceptado":
                    st.success(f"🟢 Aceptado")
                else:
                    st.error(f"🔴 Rechazado")
                
        st.markdown("---")

        if st.button("Volver al inicio"):
            volver_inicio()
            st.rerun()

    # Calculadora ADV
    elif st.session_state.pestana_activa == "Calculadora ADV":
        st.header("🔸 Calculadora ADV")
        # Inicializar variables de sesión para evitar KeyError
        if 'Decil_riesgos_guardada' not in st.session_state:
            st.session_state['Decil_riesgos_guardada'] = 0

        with st.form("form_cliente"):
            col1, col2 = st.columns(2)
            with col1:
                Solicitud = st.number_input("Solicitud", min_value=0, max_value=10000000000000, value=5000, key="Solicitud_BLMID", step=1000)

                Edad = st.number_input("Edad", min_value=18, max_value=100, step=1, key="edad_BLMID")

                InfoCre = st.selectbox("¿Cuenta con información de Crédito?", ["Si", "No"], key="InfoCre_BLMID")

                MicroScore = st.number_input("MicroScore", min_value=-9, max_value=1000, step=1, key="MicroScore_BLMID")

                ScoreFico = st.number_input("ScoreFico", min_value=0, max_value=1000, step=1, key="ScoreFico_BLMID")

                Edo_civil = st.selectbox("Estado civil", ["Casado", "Union libre", "Soltero", "Viudo", "Divorciado"], key="Edo_civil_BLMID")

                Oferta = st.number_input("Oferta", min_value=0, max_value=10000000, step=100, key="Oferta_BLMID")

                Comentarios = st.text_input("Comentario", max_chars=150, key="comentarios_BLMID", value="")

                llamada = st.selectbox("¿Se realizo llamada?", ["Si", "No"], key="LLAMADA_BLMID")

            with col2:
                genero = st.selectbox("Género", ["Masculino", "Femenino", "Otro"], key="genero_BLMID")

                Dependientes = st.selectbox("Dependiente", ["0", "1", "2", "3", "4", "5", "+5"], key="Dependientes_BLMID")

                CP = st.text_input("Código postal", max_chars=50,key = "CP",value="")
                resultadosCP = pd.DataFrame()

                if CP:
                    if len(CP) != 5:
                        st.warning("El CP debe tener al menos 5 caracteres.")
                    else:
                        tabla_CPID = CPbase()
                        tabla_CPID["d_codigo"] = tabla_CPID["d_codigo"].astype(str)
                        resultadosCP = tabla_CPID[tabla_CPID["d_codigo"] == CP]
                        if not resultadosCP.empty:
                            st.success("✅ Datos encontrados en BaseCP")
                        else:
                            st.error("No se encontraron datos para ese CP.")

                Tipo_negocio = st.selectbox("Tipo de negocio", ['ABARROTES', 'CONSUMOS', 'MISCELANEAS', 'OTROS', 'SIN INFORMACION'], key="Tipo_negocio_BLMID")

                if st.session_state.Tipo_negocio_BLMID == "SIN INFORMACION":
                    tipo_negocio_especificado = st.text_input("Especifique el tipo de negocio:", key="tipo_negocio_especificado_BLMID")

                BimboID = st.text_input("BimboID", max_chars=30, key="BimboID_BLMID", value="")

                blmId = st.text_input("blmId", max_chars=15,key = "blmId",value="")
                resultados = pd.DataFrame()

                if blmId:
                    if len(blmId) < 3:
                        st.warning("El blmId debe tener al menos 3 caracteres.")
                    else:
                        tabla_bimbo = BimboIDbase()
                        tabla_bimbo["blmId"] = tabla_bimbo["blmId"].astype(str)
                        resultados = tabla_bimbo[tabla_bimbo["blmId"] == blmId]
                        if not resultados.empty:
                            st.success("✅ Datos encontrados en Base Bimbo")
                        else:
                            st.error("No se encontraron datos para ese blmId.")

                valido_ine = st.checkbox("Validación de INE", key="INE_BLMID")
                valido_domicilio = st.checkbox("Validación de Domicilio", key="Domicilio_BLMID")
                valido_curp = st.checkbox("Validación de CURP", key="CURP_BLMID")
                valido_spei = st.checkbox("Validación de SPEI", key="SPEI_BLMID")
                valido_foto = st.selectbox("Oferta corresponde al tamaño del negocio", ["Si", "No", "No se aprecía"], key="FOTO_BLMID")

            campos_validos = (
                Edad and
                CP and len(CP) == 5 and not resultadosCP.empty and
                genero and
                Oferta and
                Dependientes and
                Edo_civil and
                (Tipo_negocio != "SIN INFORMACION" or (Tipo_negocio == "SIN INFORMACION" and 'tipo_negocio_especificado' in locals() and tipo_negocio_especificado)) and
                BimboID and
                blmId and len(blmId) >= 3 and not resultados.empty
            )

            submit_button = st.form_submit_button("Guardar registro")

            df2 = cargar_base_ADV()

            if submit_button:
                if not campos_validos:
                    st.warning("⚠️ Por favor, completa todos los campos correctamente antes de enviar.")
                elif str(Solicitud) in df2['Solicitud'].values:
                    st.warning("⚠️ Esta solicitud ya existe. Por favor, ingresa un número único.")
                else:
                    try:
                        df = cargar_base_ADV()
                        
                        # Obtener Score y Casa si se encontró el BimboID
                        Morosidad_Promedio = resultados.iloc[0]["Morosidad_Promedio"] if not resultados.empty else None
                        Gradient_Boosting_Proba = resultados.iloc[0]["Gradient Boosting_Proba"] if not resultados.empty else None
                        Decil_ventas = resultados.iloc[0]["Decil_ventas"] if not resultados.empty else None
                        PromedioVisitasXMesBimbo = resultados.iloc[0]["PromedioVisitasXMesBimbo"] if not resultados.empty else None
                        ventaPromedioSemanalUlt12Semanas = resultados.iloc[0]["ventaPromedioSemanalUlt12Semanas"] if not resultados.empty else None
                        Giro_de_Cliente = resultados.iloc[0]["Giro_de_Cliente"] if not resultados.empty else None
                        MontoMinCredito = resultados.iloc[0]["MontoMinCredito"] if not resultados.empty else None
                        DiasConCreditoVigente = resultados.iloc[0]["DiasConCreditoVigente"] if not resultados.empty else None
                        Estado = resultadosCP.iloc[0]["d_estado"] if not resultadosCP.empty else None
                        Municipio = resultadosCP.iloc[0]["D_mnpio"] if not resultadosCP.empty else None
                        Ingreso_empleado = resultadosCP.iloc[0]["Ingreso_empleado"] if not resultadosCP.empty else None
                        Morosidad = resultadosCP.iloc[0]["Morosidad"] if not resultadosCP.empty else None

                        nuevo = {
                            'Usuario_registro': str(st.session_state['usuario_actual']),
                            'Solicitud': str(Solicitud),
                            #'nombre': nombre,
                            'genero': genero,
                            'Oferta': Oferta,
                            'Informacion de Credito': InfoCre,
                            'MicroScore' : MicroScore,
                            'ScoreFico' : ScoreFico,
                            'Dependientes' : Dependientes,
                            'Estado civil' : Edo_civil,
                            'Tipo de negocio' : tipo_negocio_especificado if (Tipo_negocio == "SIN INFORMACION" and 'tipo_negocio_especificado' in locals() and tipo_negocio_especificado) else Tipo_negocio,
                            'BimboID': str(BimboID),
                            'blmId' : str(blmId),
                            'Estado de Residencia': Estado,
                            'Municipio de Residencia': Municipio,
                            'Morosidad' : Morosidad,
                            'Morosidad_Promedio': Morosidad_Promedio,
                            'Ingreso_corriente' : Ingreso_empleado,
                            'Edad': Edad,
                            'Gradient Boosting_Proba': Gradient_Boosting_Proba,
                            'Decil_riesgos': asignar_decil(ScoreFico, MicroScore),
                            'PromedioVisitasXMesBimbo': PromedioVisitasXMesBimbo,
                            'Decil_ventas': Decil_ventas,
                            'DiasConCreditoVigente': DiasConCreditoVigente,
                            'ventaPromedioSemanalUlt12Semanas': ventaPromedioSemanalUlt12Semanas,
                            'MontoMinCredito': MontoMinCredito,
                            'Giro_de_Cliente': Giro_de_Cliente,
                            'Validacion_INE' : valido_ine,
                            'Validacion_domicilio' : valido_domicilio,
                            'Validacion_curp' : valido_curp,
                            'Validacion_spei' : valido_spei,
                            'Validacion_foto' : valido_foto,
                            'Comentarios' : Comentarios,
                            'Llamada' : llamada,
                            'Fecha_registro': fecha_local
                        }

                        datos_nuevos = pd.DataFrame([nuevo])

                        rango_monto = obtener_rango_monto_adv(Oferta)

                        Decil_riesgos = asignar_decil(ScoreFico, MicroScore)

                        Tasa = obtener_porcentaje(rango_monto, Decil_riesgos)

                        Desiscion = asignar_desiscion_ADV(Decil_riesgos)

                        datos_nuevos["Decil_riesgos"] = Decil_riesgos

                        datos_nuevos["Rango_monto"] = rango_monto

                        datos_nuevos["Tasa"] = Tasa

                        datos_nuevos["Desicion"] = Desiscion

                        df = pd.concat([df, datos_nuevos], ignore_index=True)
                        guardar_base_ADV(df)
                        st.success("✅ Registro guardado correctamente en AWS_S3.")

                        # Guardamos la info en el estado de sesión para mostrar después
                        st.session_state['mostrar_resultado'] = True
                        st.session_state['solicitud_guardada_ADV'] = int(Solicitud)
                        st.session_state['blmId_guardado_ADV'] = str(blmId)
                        st.session_state['Decil_riesgos_guardada_ADV'] = int(Decil_riesgos)
                        st.session_state['Desicion_guardada_ADV'] = str(Desiscion)
                        st.session_state['Oferta_input_ADV'] = Oferta
                        st.session_state['Oferta_guardada_ADV'] = int(Oferta)  # Conversión explícita a entero
                        st.session_state['Tasa_guardada_ADV'] = str(Tasa)      # Conversión explícita a string
                        # Agregar bandera para indicar que se debe limpiar el formulario
                        st.session_state['limpiar_formulario'] = True
                        
                        # Recargar la página para mostrar los campos limpios
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error al guardar el registro: {e}")

        # Mostrar el contenedor con el resultado si existe
        if 'mostrar_resultado' in st.session_state and st.session_state['mostrar_resultado']:
            with st.container():
                st.markdown("### Resultado de la Solicitud")
                
                # Usar st.write en lugar de st.markdown para el contenido
                st.markdown(
                    f"""
                    <div style="color: white;">
                        <b>Solicitud:</b> {st.session_state.get('solicitud_guardada_ADV', 0)}<br>
                        <b>Decil:</b> {st.session_state.get('Decil_riesgos_guardada_ADV', 0)}<br>
                        <b>blmId:</b> {st.session_state.get('blmId_guardado_ADV', 'N/A')}<br>
                        <b>Tasa:</b> {st.session_state.get('Tasa_guardada_ADV', 'N/A')}<br>
                        <b>Oferta:</b> ${int(st.session_state.get('Oferta_input_ADV', '0')):,.0f}<br>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # Mostrar interpretación visual de la probabilidad
                Desicion_value = st.session_state['Desicion_guardada_ADV']
                if Desicion_value == "Aceptado":
                    st.success(f"🟢 Aceptado")
                else:
                    st.error(f"🔴 Rechazado")

        st.markdown("---")

        if st.button("Volver al inicio"):
            volver_inicio()
            st.rerun()

    elif st.session_state.pestana_activa == "Calculadora SIN BLMID":
        st.header("🔹 SIN BIMBOID")
        with st.form("form_cliente"):
            col1, col2 = st.columns(2)
            with col1:
                Solicitud = st.number_input("Solicitud", min_value=4000, max_value=10000000000000, key="Solicitud", step=1000)

                nombre = st.text_input("Nombre completo", max_chars=50, key = "nombre",value="")
                if nombre and len(nombre) < 3:
                    st.warning("El nombre debe tener al menos 3 caracteres.")

                Edad = st.number_input("Edad", min_value=18, max_value=100, step=1,key = "edad")

                InfoCre = st.selectbox("¿Cuenta con información de Crédito?", ["Si", "No"], key="InfoCre")

                MicroScore = st.number_input(
                    "MicroScore",
                    min_value=-9,
                    max_value=1000,
                    step=1,
                    key="MicroScore"
                )

                ScoreFico = st.number_input("ScoreFico", 
                    min_value=0, 
                    max_value=1000, 
                    step=1, 
                    key="ScoreFico"
                )

                Edo_civil = st.selectbox("Estado civil", ["Casado", "Union libre", "Soltero", "Viudo","Divorciado"],key = "Edo_civil")


                Oferta = st.number_input("Oferta", min_value=0, max_value=10000000, step=100,key = "Oferta")

                Comentarios = st.text_input("Comentario", max_chars=150, key = "comentarios",value="")

                llamada = st.selectbox("¿Se realizo llamada?", ["Si", "No"],key = "llamada")

            with col2:
                genero = st.selectbox("Género", ["Masculino", "Femenino", "Otro"],key = "genero")

                Dependientes = st.selectbox("Dependiente", ["0", "1", "2", "3", "4", "5", "+5"],key = "Dependientes")

                CP = st.text_input("Código postal", max_chars=50,key = "CP",value="")
                resultadosCP = pd.DataFrame()

                if CP:
                    if len(CP) != 5:
                        st.warning("El CP debe tener al menos 5 caracteres.")
                    else:
                        tabla_CPID = CPbase()
                        tabla_CPID["d_codigo"] = tabla_CPID["d_codigo"].astype(str)
                        resultadosCP = tabla_CPID[tabla_CPID["d_codigo"] == CP]
                        if not resultadosCP.empty:
                            st.success("✅ Datos encontrados en BaseCP")
                        else:
                            st.error("No se encontraron datos para ese CP.")

                Tipo_negocio = st.selectbox("Tipo de negocio", 
                                ['ABARROTES','CONSUMOS','MISCELANEAS','OTROS','SIN INFORMACION'],
                                key="Tipo_negocio",
                                on_change=None)  # Asegura que se actualice la interfaz

                if st.session_state.Tipo_negocio == "SIN INFORMACION":
                    tipo_negocio_especificado = st.text_input("Especifique el tipo de negocio:", 
                                                            key="tipo_negocio_especificado")

                blmId = st.text_input("blmId", max_chars=15,key = "blmId",value="")
                resultados = pd.DataFrame()

                if blmId:
                    if len(blmId) < 3:
                        st.warning("El blmId debe tener al menos 3 caracteres.")
                    else:
                        tabla_bimbo = BimboIDbase()
                        tabla_bimbo["blmId"] = tabla_bimbo["blmId"].astype(str)
                        resultados = tabla_bimbo[tabla_bimbo["blmId"] == blmId]
                        if not resultados.empty:
                            st.success("✅ Datos encontrados en Base Bimbo")
                        else:
                            st.error("No se encontraron datos para ese blmId.")

                valido_ine = st.checkbox("Validación de INE" , key="INE")
                valido_domicilio = st.checkbox("Validación de Domicilio", key="Domicilio")
                valido_curp = st.checkbox("Validación de CURP", key="CURP")
                valido_spei = st.checkbox("Validación de SPEI", key="SPEI")
                Hist_cred = st.checkbox("Revisión de reporte de historial crediticio", key="RHC")
                valido_foto = st.selectbox("Oferta corresponde al tamaño del negocio", ["Si", "No", "No se aprecía"],key = "FOTO")

                # Validaciones de campos obligatorios
                campos_validos = (
                    nombre and len(nombre) >= 3 and
                    Edad and
                    CP and len(CP) == 5 and not resultadosCP.empty and
                    genero and
                    Oferta and
                    Dependientes and
                    Hist_cred and
                    Edo_civil and
                    (Tipo_negocio != "SIN INFORMACION" or (Tipo_negocio == "SIN INFORMACION" and 'tipo_negocio_especificado' in locals() and tipo_negocio_especificado)) and
                    blmId and len(blmId) >= 3 and not resultados.empty
                )
                

            submit_button = st.form_submit_button("Guardar registro")

            df1 = cargar_base_SBLMID()

            Blmid_validos = BLMIDVAL()
            
            if submit_button:
                if not campos_validos:
                    st.warning("⚠️ Por favor, completa todos los campos correctamente antes de enviar.")
                elif str(Solicitud) in df1['Solicitud'].values:
                    st.warning("⚠️ Esta solicitud ya existe. Por favor, ingresa un número único.")
                else:
                    try:
                        df = cargar_base_SBLMID()
                        
                        # Obtener Score y Casa si se encontró el BimboID
                        Morosidad_Promedio = resultados.iloc[0]["Morosidad_Promedio"] if not resultados.empty else None
                        Gradient_Boosting_Proba = resultados.iloc[0]["Gradient Boosting_Proba"] if not resultados.empty else None
                        Decil_ventas = resultados.iloc[0]["Decil_ventas"] if not resultados.empty else None
                        PromedioVisitasXMesBimbo = resultados.iloc[0]["PromedioVisitasXMesBimbo"] if not resultados.empty else None
                        ventaPromedioSemanalUlt12Semanas = resultados.iloc[0]["ventaPromedioSemanalUlt12Semanas"] if not resultados.empty else None
                        Giro_de_Cliente = resultados.iloc[0]["Giro_de_Cliente"] if not resultados.empty else None
                        MontoMinCredito = resultados.iloc[0]["MontoMinCredito"] if not resultados.empty else None
                        DiasConCreditoVigente = resultados.iloc[0]["DiasConCreditoVigente"] if not resultados.empty else None
                        Estado = resultadosCP.iloc[0]["d_estado"] if not resultadosCP.empty else None
                        Municipio = resultadosCP.iloc[0]["D_mnpio"] if not resultadosCP.empty else None
                        Ingreso_empleado = resultadosCP.iloc[0]["Ingreso_empleado"] if not resultadosCP.empty else None
                        Morosidad = resultadosCP.iloc[0]["Morosidad"] if not resultadosCP.empty else None

                        nuevo = {
                            'Usuario_registro': str(st.session_state['usuario_actual']),
                            'Solicitud':int(Solicitud),
                            'nombre': nombre,
                            'genero': genero,
                            'Oferta': Oferta,
                            'Informacion de Credito': InfoCre,
                            'MicroScore' : MicroScore,
                            'ScoreFico' : ScoreFico,
                            'Dependientes' : Dependientes,
                            'Estado civil' : Edo_civil,
                            'Tipo de negocio' : tipo_negocio_especificado if (Tipo_negocio == "SIN INFORMACION" and 'tipo_negocio_especificado' in locals() and tipo_negocio_especificado) else Tipo_negocio,
                            'blmId' : str(blmId),
                            'Estado de Residencia': Estado,
                            'Municipio de Residencia': Municipio,
                            'Morosidad' : Morosidad,
                            'Morosidad_Promedio': Morosidad_Promedio,
                            'Ingreso_corriente' : Ingreso_empleado,
                            'Edad': Edad,
                            'Gradient Boosting_Proba': Gradient_Boosting_Proba,
                            'Decil_riesgos': asignar_decil(ScoreFico, MicroScore),
                            'PromedioVisitasXMesBimbo': PromedioVisitasXMesBimbo,
                            'Decil_ventas': Decil_ventas,
                            'DiasConCreditoVigente': DiasConCreditoVigente,
                            'ventaPromedioSemanalUlt12Semanas': ventaPromedioSemanalUlt12Semanas,
                            'MontoMinCredito': MontoMinCredito,
                            'Giro_de_Cliente': Giro_de_Cliente,
                            'Validacion_INE' : valido_ine,
                            'Validacion_domicilio' : valido_domicilio,
                            'Validacion_curp' : valido_curp,
                            'Validacion_spei' : valido_spei,
                            'Validacion_foto' : valido_foto,
                            'Comentarios' : Comentarios,
                            'Llamada' : llamada,
                            'Fecha_registro': fecha_local
                        }

                        datos_nuevos = pd.DataFrame([nuevo])

                        # Preprocesar los datos para calificarlos con el modelo de regresion
                        datos_preprocesados = preprocesar_nuevos_datos(datos_nuevos, modelo_cargado)
                        
                        datos_preprocesados['const'] = 1  # Añadir una constante para el intercepto

                        try:
                            probabilidades = predecir_probabilidades(datos_preprocesados, modelo_cargado)
                            st.success("✅ Modelo Interno ejecutado correctamente.")
                        except Exception as e:
                            st.warning("⚠️ No se pudo ejecutar el modelo. Se usará un valor por defecto del 55%.")
                            probabilidades = [0.35591865]  # Valor por defecto

                        Decil = obtener_decil(probabilidades[0])

                        datos_nuevos["Probabilidad"] = probabilidades
                        datos_nuevos["Decil_modelo"] = Decil

                        Desiscion = asignar_desiscion_SBLMID(Decil)

                        datos_nuevos['Desiscion'] = Desiscion

                        print(Desiscion)
                    
                        df = pd.concat([df, datos_nuevos], ignore_index=True)
                        guardar_base_SBLMID(df)
                        st.success("✅ Registro guardado correctamente en AWS_S3.")

                        # Guardamos la info en el estado de sesión para mostrar después
                        st.session_state['mostrar_resultado'] = True
                        st.session_state['solicitud_guardada'] = int(Solicitud)
                        st.session_state['nombre_guardado'] = nombre
                        st.session_state['blmId_guardado'] = str(blmId)
                        st.session_state['Decil_guardado'] = str(Decil)
                        st.session_state['Desicion_guardada'] = str(Desiscion)
                        st.session_state['Oferta_input'] = str(Oferta)
                        st.session_state['probamodelo1_guardado'] = float(probabilidades[0]*100)
                        
                        # Agregar bandera para indicar que se debe limpiar el formulario
                        st.session_state['limpiar_formulario'] = True
                        
                        # Recargar la página para mostrar los campos limpios
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error al guardar el registro: {e}")

        # Mostrar el contenedor con el resultado si existe
        if 'mostrar_resultado' in st.session_state and st.session_state['mostrar_resultado']:
            with st.container():
                st.markdown("### Resultado de la Solicitud")
                
                # Mostrar resultado con fondo transparente y letra blanca usando HTML
                st.markdown(
                    f"""
                    <div style="color: white;">
                        <b>Solicitud:</b> {st.session_state.get('solicitud_guardada', 0)}<br>
                        <b>Decil:</b> {st.session_state.get('Decil_guardado', 'N/A')}<br>
                        <b>blmId:</b> {st.session_state.get('blmId_guardado', 'N/A')}<br>
                        <b>Oferta:</b> ${int(st.session_state.get('Oferta_input', '0')):,.0f}<br>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # Mostrar interpretación visual de la probabilidad
                Desicion_value = st.session_state['Desicion_guardada']
                if Desicion_value == "Aceptado":
                    st.success(f"🟢 Aceptado")
                else:
                    st.error(f"🔴 Rechazado")
                
        st.markdown("---")

        if st.button("Volver al inicio"):
            volver_inicio()
            st.rerun()


    # Cerrar sesión
    if st.sidebar.button("Cerrar Sesión"):
        st.session_state['autenticado'] = False
        st.session_state['usuario_actual'] = ""
        st.session_state['pestana_activa'] = "inicio"
        st.rerun()
