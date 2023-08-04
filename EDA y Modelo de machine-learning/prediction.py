import main
import joblib
import pandas as pd
import tensorflow as tf
import data
from datetime import datetime
from pymongo.errors import PyMongoError
import time

# Cargo la respuesta
def convert_to_df(nueva_respuesta):
    try:
        nueva_rta_df = pd.DataFrame(nueva_respuesta)
    except Exception as e:
        print("Error al tratar de convertir los datos a dataframe:", e)
        nueva_rta_df = pd.DataFrame()  
    return nueva_rta_df


# Selecciono el usuario
def select_user(df):
    try:
        user = df['id_de_slack'].values[0]
    except Exception as e:
        print("Error al seleccionar un usuario:", e)
        user = None       
    return user


# Selecciono el modelo del usuario
def select_user_model(user):
    try:
        nombre_del_modelo = 'model_' + user
    except Exception as e:
        print("Error al tratar de seleccionar el modelo del usuario:", e)
        nombre_del_modelo = 'model_'
    return nombre_del_modelo


#Selecciono la direccion del modelo 
direccion = 'C://Users//agusv//' # -------- CAMBIAR SEGUN LA PC ---------------------------------------------------
def select_model_dir(model_name):
    try:
        direccion_model = direccion + model_name
    except Exception as e:
        print("Error al seleccionar la dirección del modelo:", e)
        direccion_model = None
    return direccion_model


# Selecciono el modelo
def select_model(model_dir):
    try:
        model = joblib.load(str(model_dir))
    except Exception as e:
        print("Error al seleccionar el modelo:", e)
        model = None
    return model


# Obtener las características y etiquetas
def df_y(df):
    try:
        y_df_nuevo = df['productividad_hoy']
    except Exception as e:
        print("Error al obtener nuevo df_y:", e)
        y_df_nuevo = None
    return y_df_nuevo


def df_x(df):
    try:
        columnas_a_dropear = ['_id', 'fecha_hora', 'id_de_slack', 'nombre_usuario_slack', 'nombre_completo', 'con_que_areas_te_vas_a_reunir', 'productividad_hoy']
        x_df_nuevo = df.drop(columnas_a_dropear, axis=1)
    except Exception as e:
        print("Error al obtener nuevo df_x:", e)
        x_df_nuevo = None
    return x_df_nuevo


def get_dummies_df_x(x_df_nuevo):
    try:
        x_df_dummies = pd.get_dummies(x_df_nuevo, columns = [
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
                    ])
    except Exception as e:
        print("Error al transoformar df_x a dummies:", e)
        x_df_dummies = None
    return x_df_dummies


# Lista de columnas en dfeval
columnas_dfeval = ['tienes_agenda_planeada_No', 'tienes_agenda_planeada_Si',
        'tienes_reuniones_planeadas_No', 'tienes_reuniones_planeadas_Si',
        'ayunaste_hoy_No', 'ayunaste_hoy_Si', 'tienes_deadline_hoy_No',
        'tienes_deadline_hoy_Si', 'alguna_tarea_te_llevo_mas_tiempo_No',
        'alguna_tarea_te_llevo_mas_tiempo_Si',
        'hay_tareas_que_se_pueden_automatizar_No',
        'hay_tareas_que_se_pueden_automatizar_Si',
        'tuviste_que_ir_a_algun_lugar_para_hacer_tus_tareas_No',
        'tuviste_que_ir_a_algun_lugar_para_hacer_tus_tareas_Si',
        'usaste_alguna_metodologia_para_optimizar_el_tiempo_No',
        'usaste_alguna_metodologia_para_optimizar_el_tiempo_Si',
        'fueron_satisfactorias_las_reuniones_No',
        'fueron_satisfactorias_las_reuniones_No tuve ninguna reunión',
        'fueron_satisfactorias_las_reuniones_Si',
        'pudiste_resolver_tus_dudas_sobre_el_trabajo_No',
        'pudiste_resolver_tus_dudas_sobre_el_trabajo_No tuve ninguna reunion',
        'pudiste_resolver_tus_dudas_sobre_el_trabajo_Si',
        'pudiste_resolver_tus_dudas_sobre_el_trabajo_Tuve reuniones, pero, no tenía dudas por aclarar',
        'tuviste_reuniones_planificadas_No',
        'tuviste_reuniones_planificadas_Si',
        'calificacion_descansos_Insuficientes pero bien distribuidos',
        'calificacion_descansos_Insuficientes y mal distribuidos',
        'calificacion_descansos_Suficientes pero mal distribuidos',
        'calificacion_descansos_Suficientes y bien distribuidos']


def predict(nueva_respuesta):

    # Cargo la respuesta
    nueva_rta_df = convert_to_df(nueva_respuesta)

    # Selecciono el usuario
    user = select_user(nueva_rta_df)

    # Selecciono el modelo del usuario
    nombre_del_modelo = select_user_model(user)

    #Selecciono la direccion del modelo 
    model_dir = f'models/{nombre_del_modelo}'
    
    # Selecciono el modelo
    modelo = select_model(model_dir)

    # Obtener las características y etiquetas
    y_df_nuevo = df_y(nueva_rta_df)
    x_df_nuevo = df_x(nueva_rta_df)

    # df_x dummies
    x_df_nuevo = get_dummies_df_x(x_df_nuevo)

    # Asegurarse que x_df_nuevo tenga las mismas columnas que dfeval y en el mismo orden
    x_df_nuevo = x_df_nuevo.reindex(columns=columnas_dfeval, fill_value=0)

    # Compilar el modelo
    optimizer = tf.keras.optimizers.Adam()
    loss_fn = tf.keras.losses.mean_squared_error
    modelo.compile(optimizer=optimizer, loss=loss_fn)

    # Entrenar el modelo con los nuevos datos
    modelo.fit(x_df_nuevo, y_df_nuevo, epochs=10, batch_size=32)

    # Hacer la prediccion
    prediction = int(modelo.predict(x_df_nuevo))
    
    return prediction


def save_last_prediction(user_id, prediction):
    try:
        db = data.connect_to_db()
        last_date_collections = db['predictions_history']
        document = {'user_id': user_id, 'fecha_hora': datetime.now(), 'prediction': prediction}
        last_date_collections.insert_one(document)
        print("Registro insertado exitosamente.")
    except PyMongoError as e:
        print(f"Error al insertar el registro: {e}")