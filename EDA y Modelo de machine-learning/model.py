from pymongo import MongoClient
import pandas as pd
from datetime import datetime
import tensorflow as tf
import os
import joblib

# Definir constantes
EPOCHS = 32
BATCH_SIZE = 32
INPUT_SHAPE = (29,)

CATEGORICAL_COLUMNS = [
    'tienes_agenda_planeada', 'tienes_reuniones_planeadas',
    'ayunaste_hoy', 'tienes_deadline_hoy',
    'alguna_tarea_te_llevo_mas_tiempo',
    'hay_tareas_que_se_pueden_automatizar',
    'tuviste_que_ir_a_algun_lugar_para_hacer_tus_tareas',
    'usaste_alguna_metodologia_para_optimizar_el_tiempo',
    'fueron_satisfactorias_las_reuniones',
    'pudiste_resolver_tus_dudas_sobre_el_trabajo',
    'tuviste_reuniones_planificadas',
    'calificacion_descansos'
]
NUMERIC_COLUMNS = []

# Definicion de variables
# Establece los datos que se van a tomar para entrenar y evaluar el modelo


def cargar_datos(df_usuario):
    try:
        #df_usuario = df.loc[df['id_de_slack'] == user]
        total_registros = len(df_usuario)
        porcentaje = 0.7
        filtro1 = int(total_registros * porcentaje)

        dftrain = df_usuario.head(filtro1)  # Conjunto de entrenamiento
        # Conjunto de evaluación
        dfeval = df_usuario.tail(total_registros - filtro1)

        y_train = dftrain.pop('productividad_hoy')
        y_eval = dfeval.pop('productividad_hoy')

        columnas_df = set(df_usuario.columns)
        columnas_categoricas = set(CATEGORICAL_COLUMNS)
        columnas_faltantes = columnas_categoricas - columnas_df

        if columnas_faltantes:
            raise ValueError(
                f"Las siguientes columnas categóricas no se encontraron en el dataframe: {columnas_faltantes}")

        return dftrain, dfeval, y_train, y_eval

    except Exception as e:
        # Manejo de errores: imprimir el error y regresar None para indicar que ha ocurrido un error.
        print(f"Error al cargar los datos: {e}")
        return None, None, None, None, None, None


# Transforma los datos de las columnas de categoricas a numericas
def transformar_datos(dftrain, dfeval):
    try:
        dftrain = dftrain.drop('con_que_areas_te_vas_a_reunir', axis=1)
        dfeval = dfeval.drop('con_que_areas_te_vas_a_reunir', axis=1)

        #dftrain = dftrain.drop('productividad_hoy', axis=1)
        #dfeval = dfeval.drop('productividad_hoy', axis=1)

        dftrain = pd.get_dummies(dftrain, columns=CATEGORICAL_COLUMNS)
        dftrain = dftrain.drop('id_de_slack', axis=1)

        dfeval = pd.get_dummies(dfeval, columns=CATEGORICAL_COLUMNS)
        dfeval = dfeval.drop('id_de_slack', axis=1)

        return dftrain, dfeval

    except Exception as e:
        # Manejo de errores: imprimir el error y regresar None para indicar que ha ocurrido un error.
        print(f"Error al transformar los datos: {e}")
        return None, None


# Crea el modelo de regresion
def crear_modelo(dftrain, y_train):
    try:
        feature_columns = []
        for feature_name in dftrain.columns:
            feature_columns.append(tf.feature_column.numeric_column(
                feature_name, dtype=tf.float32))

        model = tf.keras.Sequential([
            tf.keras.layers.Dense(1, input_shape=(len(feature_columns),))
        ])

        model.compile(loss='mse', optimizer='sgd')

        model.fit(dftrain, y_train, epochs=32, batch_size=32, verbose=0)

        return model

    except Exception as e:
        # Manejo de errores: imprimir el error y regresar None para indicar que ha ocurrido un error.
        print(f"Error al crear el modelo: {e}")
        return None


# Guarda el modelo en el mismo directorio en el que se ejecuta el codigo
def guardar_modelo_2(model, user):
    try:
        nombre = 'model_' + user
        joblib.dump(model, str(nombre))

        # Mostrar mensaje de éxito
        print("Modelo guardado exitosamente.")

    except Exception as e:
        # Manejo de errores: imprimir el error.
        print(f"Error al guardar el modelo: {e}")


def guardar_modelo(model, user):
    try:
        # Crear el directorio 'directorio_modelos' si no existe
        if not os.path.exists('models'):
            os.makedirs('models')

        # Construir la ruta completa para el archivo de guardado
        nombre = f'models/model_{user}'

        # Guardar el modelo en la carpeta 'directorio_modelos'
        joblib.dump(model, nombre)

        # Mostrar mensaje de éxito
        print("Modelo guardado exitosamente.")

    except Exception as e:
        # Manejo de errores: imprimir el error.
        print(f"Error al guardar el modelo: {e}")


# Funcion principal que ejecuta y guarda el modelo
def ejecucion_modelo(df, user):
    try:
        dftrain, dfeval, y_train, y_eval = cargar_datos(
            df)

        if dftrain is None:
            # Ocurrió un error al cargar los datos.
            return

        dftrain, dfeval = transformar_datos(
            dftrain, dfeval)

        if dftrain is None:
            # Ocurrió un error al transformar los datos.
            return

        model = crear_modelo(dftrain, y_train)

        if model is None:
            # Ocurrió un error al crear el modelo.
            return

        guardar_modelo(model, user)

    except Exception as e:
        # Manejo de errores: imprimir el error.
        print(f"Error durante la ejecución del modelo: {e}")


def reentrenar_modelo(model, df_new_inputs, user):
    try:
        # Tomamos el último registro del DataFrame df_new_inputs
        # df_nuevo = df_new_inputs.tail(1)
        y_df_nuevo = df_new_inputs.pop('productividad_hoy')

        # Eliminamos las columnas innecesarias para el entrenamiento
        columnas_a_dropear = ['_id', 'fecha_hora', 'id_de_slack', 'nombre_usuario_slack',
                              'nombre_completo', 'con_que_areas_te_vas_a_reunir']  # adicionar 'productividad_hoy'
        x_df_nuevo = df_new_inputs.drop(columnas_a_dropear, axis=1)

        # Transformamos los datos nuevos en dummies
        x_df_nuevo = pd.get_dummies(x_df_nuevo, columns=CATEGORICAL_COLUMNS)

        # Reentrenamos el modelo con el nuevo registro
        model.compile(optimizer=tf.keras.optimizers.Adam(),
                      loss=tf.keras.losses.mean_squared_error)
        model.fit(x_df_nuevo, y_df_nuevo, epochs=10, batch_size=32, verbose=0)
        guardar_modelo(model, user)
        print("modelo reentrenado")
        return model

    except Exception as e:
        # Manejo de errores: imprimir el error y regresar None para indicar que ha ocurrido un error.
        print(f"Error al reentrenar el modelo: {e}")
        return None


def retrain_model(df_new_inputs, user):
    # Obtener el último registro de df_new_inputs
    # df_nuevo = df_new_inputs.tail(1).copy()

    # Verificar si la columna objetivo 'productividad_hoy' está presente en df_nuevo
    if 'productividad_hoy' not in df_nuevo.columns:
        # Si no está presente, copiar el valor de 'productividad_hoy' del DataFrame original df_new_inputs
        last_productivity = df_new_inputs.loc[df_new_inputs['id_de_slack']
                                              == user, 'productividad_hoy'].values[-1]
        df_nuevo['productividad_hoy'] = last_productivity

    # Ahora asegurémonos de que el DataFrame solo contenga las columnas relevantes para el modelo
    relevant_columns = ['tienes_agenda_planeada', 'tienes_reuniones_planeadas', 'ayunaste_hoy',
                        'tienes_deadline_hoy', 'alguna_tarea_te_llevo_mas_tiempo',
                        'hay_tareas_que_se_pueden_automatizar',
                        'tuviste_que_ir_a_algun_lugar_para_hacer_tus_tareas',
                        'usaste_alguna_metodologia_para_optimizar_el_tiempo',
                        'fueron_satisfactorias_las_reuniones',
                        'pudiste_resolver_tus_dudas_sobre_el_trabajo',
                        'tuviste_reuniones_planificadas', 'calificacion_descansos']

    df_nuevo = pd.get_dummies(df_nuevo, columns=relevant_columns)

    # Eliminamos la columna 'id_de_slack' ya que no es necesaria para el reentrenamiento
    df_nuevo = df_nuevo.drop('id_de_slack', axis=1)

    # Cargar el modelo previamente entrenado
    #model = tf.keras.models.load_model('model_' + user)
    nombre = 'model_' + user
    model = joblib.load(nombre)

    # Obtener las características de entrada y la variable objetivo para el reentrenamiento
    y_train = df_nuevo.pop('productividad_hoy')

    # Reentrenar el modelo usando el último registro
    model.fit(df_nuevo, y_train, epochs=10, batch_size=32)

    # Guardar el modelo reentrenado
    model.save('model_' + user)

    # Mostrar mensaje de éxito
    print(f"Modelo para el usuario {user} reentrenado correctamente.")
