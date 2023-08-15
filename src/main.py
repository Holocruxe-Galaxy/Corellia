#from sys import last_value
import data
import model
import os
from datetime import datetime
import pandas as pd
import joblib
import time


def main():
    try:
        # importar la base de datos limpia y concatenada
        db = data.data_base()
        user_data = data.user_data()
        date = datetime.now().strftime('%Y-%m-%d')

        # se itera por cada usuario registrado en la coleccion 'user_data'
        for user in user_data['id_de_slack']:
            user = str(user)
            # nombre = 'model_' + user
            nombre = f'models/model_{user}'

            # verificacion si el usuario ya tiene un modelo entrenado
            if os.path.exists(nombre):
                # reentrenar modelo
                print('Este modelo ya está generado para el usuario', user)

                # verificacion si el usuario tiene registro del dia
                # Hacer una copia del DataFrame
                input_user = db[db['id_de_slack'] == user].copy()

                # Convertir la columna 'fecha_hora' al formato '%Y-%m-%d' utilizando .dt.strftime sin asignar
                input_user['fecha_hora'] = pd.to_datetime(
                    input_user['fecha_hora']).dt.strftime('%Y-%m-%d')

                # verifica si el usuario tiene registros nuevos
                if date in input_user['fecha_hora'].values:
                    # reentrenar el modelo con los nuevos registros
                    # verificamos la ultima fecha de entrenamiento del modelo
                    last_train = data.last_train(user)
                    # tomamos los registros posteriores a la fecha
                    df_new_inputs = data.filter_new_records(input_user, last_train)
                    # reentrenamos el modelo
                    modelo = joblib.load(nombre)
                    model.reentrenar_modelo(modelo, df_new_inputs, user)

                else:
                    print('El usuario no cuenta con nuevos registros para reentrenar el modelo')

            else:
                # se debe establecer si el usuario cuenta con registros suficientes
                # Hacer una copia del DataFrame
                input_user = db[db['id_de_slack'] == user].copy()

                # verificar si el usuario tiene más de 30 registros
                if input_user.shape[0] >= 30:
                    # entrenamiento y guardar el nuevo modelo
                    columnas = ['_id', 'fecha_hora', 'nombre_usuario_slack', 'nombre_completo']
                    input_user.drop(columnas, axis=1, inplace=True)
                    model.ejecucion_modelo(input_user, user)

                else:
                    # Hacer la verificacion si el área cuenta con más de 30 registros
                    user_area = user_data.loc[user_data['id_de_slack'] == user]['area']
                    # Identificar usuarios de la misma área
                    area_value = user_area.iloc[0] if not user_area.empty else None
                    # Identifica cantidad de registros de usuarios de la misma área
                    users_area = user_data.loc[user_data['area'] == area_value, 'id_de_slack'].tolist()
                    data_total_area = db[db['id_de_slack'].isin(users_area)]
                    # Verifica si la data de los usuarios de la misma área es suficiente
                    if data_total_area.shape[0] >= 30:
                        # Entrena y guarda el modelo con los datos del área.
                        columnas = ['_id', 'fecha_hora', 'nombre_usuario_slack', 'nombre_completo']
                        data_total_area.drop(columnas, axis=1, inplace=True)
                        model.ejecucion_modelo(data_total_area, user)
                    else:
                        print(f'No existen suficientes datos para generar una predicción para el usuario {user}')

    except Exception as e:
        print('Ocurrió un error:', e)

if __name__ == "__main__":
    main()
