from pymongo import MongoClient
import pandas as pd
from datetime import datetime
import tensorflow as tf
from pymongo.errors import PyMongoError
import time
#import os

# Definir columnas categóricas
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

# Conexión a la base de datos MongoDB
db_connection = None


def connect_to_db():
    global db_connection

    # Si ya hay una conexión, devolverla directamente
    if db_connection is not None:
        return db_connection

    try:
        # Intentar establecer una conexión a la base de datos
        client = MongoClient('localhost', 27017)
        db = client.holobot_database_model_3

        # Asignar la conexión a la variable global
        db_connection = db

        return db
    except Exception as e:
        # Capturar cualquier excepción que pueda ocurrir durante la conexión
        # Puedes personalizar el manejo de errores según tus necesidades
        print(f"Error al conectar a la base de datos: {e}")
        return None


# Recuperar los documentos de la coleccion
def get_collection_data(db, collection_name):
    try:
        collection = db[collection_name]
        data = collection.find()
        df = pd.DataFrame.from_records(data)
        return df
    except Exception as e:
        print(
            f"Error al obtener datos de la colección {collection_name}: {str(e)}")
        return None


def get_data_from_db(collection_name):
    try:
        # Conectar a la base de datos
        db = connect_to_db()
        # Obtener los datos de la colección especificada
        df = get_collection_data(db, collection_name)
        # Convertir columna 'fecha_hora' a tipo datetime
        df['fecha'] = pd.to_datetime(df['fecha_hora']).dt.date

        return df

    except Exception as e:
        # Manejo de errores: imprimir el error y regresar un DataFrame vacío
        print(f"Error al obtener los datos de la base de datos: {e}")
        return pd.DataFrame()


# Insertar fecha de último entrenamiento

def insert_last_date(user_id):
    try:
        db = connect_to_db()
        last_date_collections = db['date_last_training']
        document = {'user_id': user_id, 'fecha_hora': datetime.now()}
        last_date_collections.insert_one(document)
        print("Registro insertado exitosamente.")
    except PyMongoError as e:
        print(f"Error al insertar el registro: {e}")


# Filtrar los registros creados después de la fecha de referencia
def filter_new_records(df, fecha_ultimo_entrenamiento):
    try:
        # Convertir la fecha de último entrenamiento a formato datetime
        fecha_ultimo_entrenamiento = pd.to_datetime(fecha_ultimo_entrenamiento)
        # Eliminar duplicados basados en la columna 'fecha_hora'
        df = df.drop_duplicates(subset='fecha_hora')
        # Reiniciar el índice del DataFrame
        df.reset_index(drop=True, inplace=True)
        # Convertir la columna 'fecha_hora' a formato datetime
        df['fecha_hora'] = pd.to_datetime(df['fecha_hora'])
        # Filtrar los nuevos registros basados en la fecha de último entrenamiento
        df_nuevos_registros = df[df['fecha_hora'] > fecha_ultimo_entrenamiento]

        return df_nuevos_registros

    except Exception as e:
        # Si ocurre alguna excepción, imprimir el mensaje de error y retornar un DataFrame vacío
        print(f"Error en la función: {e}")
        return pd.DataFrame()


# Funcion general para el procesamiento de la informacion
# Entrega un dataframe concatenando las respuestas de los 2 formularios y eliminando las columnas redundantes
def data_base():
    try:
        # Conectar a la base de datos utilizando un contexto (with)
        db = connect_to_db()
        # Obtener los datos de las colecciones
        db = connect_to_db()
        df_initial_answer = get_collection_data(db, 'initial_questions')
        df_final_answer = get_collection_data(db, 'final_questions')
        # Convertir columna 'fecha_hora' a tipo datetime en ambos DataFrames
        df_initial_answer['fecha'] = pd.to_datetime(
            df_initial_answer['fecha_hora']).dt.date
        df_final_answer['fecha'] = pd.to_datetime(
            df_final_answer['fecha_hora']).dt.date

        # Dropear columnas innecesarias en df_final_answer
        df_final_answer = df_final_answer.drop(
            columns=['_id', 'fecha_hora', 'nombre_usuario_slack', 'nombre_completo'])

        # Concatenar los dataframes
        df = pd.merge(df_initial_answer, df_final_answer,
                      on=['id_de_slack', 'fecha'])

        # Dropear la columna 'fecha' (ya se ha utilizado para hacer el merge)
        df = df.drop(columns=['fecha'])

        return df

    except Exception as e:
        # Manejo de errores: imprimir el error y regresar un DataFrame vacío
        print(f"Error al obtener los datos de la base de datos: {e}")
        return pd.DataFrame()


# Funcion para la obtencion de la ultima fecha de entrenamiento
def last_train(user_id):
    db = connect_to_db()
    # Reemplaza 'nombre_de_tu_coleccion' con el nombre real de tu colección en MongoDB
    collection = db['date_last_training']
    # Aquí, utiliza los métodos de PyMongo para acceder a los datos en la colección
    document = collection.find_one({'user_id': user_id})

    if document is not None:
        fecha_ultimo_entrenamiento = document['fecha_hora']
        fecha_ultimo_entrenamiento = fecha_ultimo_entrenamiento.strftime(
            '%Y-%m-%d')
        return fecha_ultimo_entrenamiento
    else:
        # Manejo si no se encuentra ningún documento con el 'user_id' dado
        fecha_cero = '2023-01-01'
        return fecha_cero

# Función para la obtención de los registros de la colección user_data


def user_data():
    try:
        # Obtener la conexión a la base de datos
        db = connect_to_db()
        # Obtener los datos de la colección 'user_data'
        # Reemplaza 'get_collection_data' con tu función para obtener datos
        df_users_info = get_collection_data(db, 'user_data')
        # print(df_users_info.columns)
        return df_users_info

    except Exception as e:
        print("Error al obtener los datos de la base de datos:", e)
        return pd.DataFrame()  # Devuelve un DataFrame vacío en caso de error
