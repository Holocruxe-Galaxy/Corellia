import prediction
import data
import pandas as pd
from flask import Flask, request
from pymongo import MongoClient
import pymongo

# Creacion de la instancia de la clase Flask
app = Flask(__name__)


@app.route('/')
def index():
    mensaje = "Bienvenido a Holobot. Para ingresar un nuevo dato vaya a /query. Para consultar un usuario vaya a /users."
    return mensaje


@app.route('/query', methods=['GET'])
def query():
    # Conexión a MongoDB
    client = MongoClient('mongodb://localhost:27017/')
    db = client['holobot_db_prueba']
    input_de_datos_colection = db['input_de_datos']
    output_de_datos_colection = db['output_de_datos']

    # Insertamos documento de respuestas iniciales
    def input_de_datos(respuesta):
        document = {
            'respuesta': respuesta
        }
        input_de_datos_colection.insert_one(document)

    # Insertamos documento de respuestas iniciales
    def output_de_datos(respuesta):
        document = {
            'respuesta': respuesta
        }
        output_de_datos_colection.insert_one(document)

    #http://127.0.0.1:5000/query?respuesta=

    # Obtenemos la respuesta de la URL con el metodo .get() 
    respuesta_str = request.args.get('respuesta')
    print(f"Respuesta recibida: {respuesta_str}") 
    
    # Convertimos la respuesta de cadena a diccionario, di la cadena no tiene el formato correcto de un diccionario, eval() arrojará un error, y el código dentro del bloque except se ejecutara
    try:
        respuesta_dict = eval(respuesta_str)
    except Exception as e:
        print(f"Error al evaluar: {e}")  
        return "Error al convertir la respuesta a formato de diccionario."


    # Utilizamos el modelo de predicción para obtener la productividad predicha
    try:
        predicted_productivity = prediction.predict(respuesta_dict)
    except Exception as e:
        return f"Error al predecir la productividad: {str(e)}"

    # Almacenamos la respuesta original en input_de_datos
    input_de_datos(respuesta_str)

    # Almacenamos la productividad predicha en output_de_datos
    output_de_datos(str(predicted_productivity))
    
    return str(predicted_productivity)


# Pruebas para devolver un df de un usuario X con Flask. 
@app.route('/users')
def users():
    new_user = request.args 
    new_user = str(new_user)
    db = data.data_base()
    try:
        db_user =  db[db['id_de_slack'] == new_user]
        db_user_dict = db_user.to_dict()
        return db_user_dict
    except:
        mensaje = "No exiset ese usuario"
        return mensaje
    # Ejemplo de usuario: slack_user0

    # Ejemplo de url http://127.0.0.1:5000/users?user=slack_user0

if __name__ == '__main__':
    app.run(debug=True)
