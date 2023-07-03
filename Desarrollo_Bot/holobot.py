######################### IMPORTACION DE LIBRERIAS NECESARIAS################
from datetime import datetime, timedelta
from unittest import result
import tensorflow as tf
from flask import Flask, request, jsonify
from slackeventsapi import SlackEventAdapter
from dotenv import load_dotenv
import os
import time
import datetime
from pathlib import Path
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import pymongo
from pymongo import MongoClient
import schedule
import json
from werkzeug.serving import make_server
import signal
import threading
import tensorflow as tf
import emoji

########################### CONEXIÓN CON LA BASE DE DATOS MONGO ###################
# Validar la existencia de documentos para preguntas iniciales y finales por usuario para evitar duplicados
# Conexión a MongoDB
client = MongoClient('mongodb://localhost:27017/')

# Crear la base de datos 'holobot_database'
db = client['holobot_database']

# Crear la colección 'initial_questions'
initial_questions_collection = db['initial_questions']

# Crear la colección 'final_questions'
final_questions_collection = db['final_questions']

################### CONFIGURACION PARA LA INTERACCION DE LA API ################
# Ruta de acceso al archivo '.env' para acceder al token de la app
env_path = Path('.') / '.env'

# Se carga la variable de entorno
load_dotenv(dotenv_path=env_path)

# Se crea una instancia que represnta la app
app = Flask(__name__)

# manejador de eventos de flask
slack_event_adapter = SlackEventAdapter(
    os.environ['SIGNING_SECRET'], '/slack/events', app)

# Se crea una instancia para interactuar con la API
client = WebClient(token=os.environ['SLACK_TOKEN'])
BOT_ID = client.api_call("auth.test")['user_id']

######################### CREACION DE COLECCIONES EN LA BASE DE DATOS - MONGO ################
# Funciones de inserción de datos en las bases de datos


def insert_initial_questions(user_id, slack_username, full_name, answers):
    document = {
        # Fecha y hora en que se completa la encuesta
        'fecha_hora': datetime.datetime.now(),
        'id_de_slack': user_id,
        'nombre_usuario_slack': slack_username,
        'nombre_completo': full_name,
        **answers
    }
    initial_questions_collection.insert_one(document)

# Insertar documento de preguntas finales


def insert_final_questions(user_id, slack_username, full_name, answers):
    document = {
        'fecha_hora': datetime.datetime.now(),
        'id_de_slack': user_id,
        'nombre_usuario_slack': slack_username,
        'nombre_completo': full_name,
        **answers
    }
    final_questions_collection.insert_one(document)

############################ CREACIÓN DE FORMULARIOS ########################
# Definición de preguntas - formulario de finalización de jornada con botones y menús desplegables

# Formulario de la tarde


def enviar_formulario_tt(user_id):
    formulario = {
        "blocks": [
            {
                "type": "header",
                        "text": {
                            "type": "plain_text",
                            "text": emoji.emojize(":robot_face:") + " Formulario de finalización de jornada " + emoji.emojize(":robot_face:"),
                            "emoji": True
                        }
            },
            {
                "type": "section",
                        "text": {
                            "type": "plain_text",
                            "text": "Responda las siguientes preguntas, esta información me será útil para ayudarte a optimizar tus tiempos y no te tomará mucho tiempo. ",
                            "emoji": True
                        }
            },
            {
                "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": emoji.emojize(":white_check_mark:") + " ¿Alguna tarea desarrollada le llevo más tiempo del estimado para la misma? " + emoji.emojize(":clock1:")
                        },
                "accessory": {
                            "type": "radio_buttons",
                            "options": [
                                {
                                    "text": {
                                        "type": "plain_text",
                                        "text": "Si",
                                        "emoji": True
                                    },
                                    "value": "value-0"
                                },
                                {
                                    "text": {
                                        "type": "plain_text",
                                        "text": "No",
                                        "emoji": True
                                    },
                                    "value": "value-1"
                                }
                            ],
                            "action_id": "radio_buttons-action_t1"
                        }
            },
            {
                "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": emoji.emojize(":white_check_mark:") + " Dentro de las tareas desarrolladas, ¿considera que hay algunas que se podrían automatizar para ayudarle a ahorrar tiempo? " + emoji.emojize(":robot_face:")
                        },
                "accessory": {
                            "type": "radio_buttons",
                            "options": [
                                {
                                    "text": {
                                        "type": "plain_text",
                                        "text": "Si",
                                        "emoji": True
                                    },
                                    "value": "value-0"
                                },
                                {
                                    "text": {
                                        "type": "plain_text",
                                        "text": "No",
                                        "emoji": True
                                    },
                                    "value": "value-1"
                                }
                            ],
                            "action_id": "radio_buttons-action_t2"
                        }
            },
            {
                "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": emoji.emojize(":white_check_mark:") + " ¿Tuviste que desplazarte físicamente para poder llevar a cabo tus tareas? " + emoji.emojize(":runner:")
                        },
                "accessory": {
                            "type": "radio_buttons",
                            "options": [
                                {
                                    "text": {
                                        "type": "plain_text",
                                        "text": "Si",
                                        "emoji": True
                                    },
                                    "value": "value-0"
                                },
                                {
                                    "text": {
                                        "type": "plain_text",
                                        "text": "No",
                                        "emoji": True
                                    },
                                    "value": "value-1"
                                }
                            ],
                            "action_id": "radio_buttons-action_t3"
                        }
            },
            {
                "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": emoji.emojize(":white_check_mark:") + " ¿Hiciste uso de alguna metodología enfocada en la optimización del tiempo en el desarrollo de tus tareas? " + emoji.emojize(":alarm_clock:")
                        },
                "accessory": {
                            "type": "radio_buttons",
                            "options": [
                                {
                                    "text": {
                                        "type": "plain_text",
                                        "text": "Si",
                                        "emoji": True
                                    },
                                    "value": "value-0"
                                },
                                {
                                    "text": {
                                        "type": "plain_text",
                                        "text": "No",
                                        "emoji": True
                                    },
                                    "value": "value-1"
                                }
                            ],
                            "action_id": "radio_buttons-action_t4"
                        }
            },
            {
                "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": emoji.emojize(":white_check_mark:") + " ¿Tuviste alguna reunión que no estuviera previamente planificada? " + emoji.emojize(":alarm_clock:")
                        },
                "accessory": {
                            "type": "radio_buttons",
                            "options": [
                                {
                                    "text": {
                                        "type": "plain_text",
                                        "text": "Si",
                                        "emoji": True
                                    },
                                    "value": "value-0"
                                },
                                {
                                    "text": {
                                        "type": "plain_text",
                                        "text": "No",
                                        "emoji": True
                                    },
                                    "value": "value-1"
                                }
                            ],
                            "action_id": "radio_buttons-action_t5"
                        }
            },
            {
                "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": emoji.emojize(":white_check_mark:") + " Si tuviste reuniones durante tu jornada, ¿resultaron satisfactorias en términos de cumplimiento de los objetivos? " + emoji.emojize(":thumbsup:")
                        },
                "accessory": {
                            "type": "radio_buttons",
                            "options": [
                                {
                                    "text": {
                                        "type": "plain_text",
                                        "text": "Si",
                                        "emoji": True
                                    },
                                    "value": "value-0"
                                },
                                {
                                    "text": {
                                        "type": "plain_text",
                                        "text": "No",
                                        "emoji": True
                                    },
                                    "value": "value-1"
                                },
                                {
                                    "text": {
                                        "type": "plain_text",
                                        "text": "No tuve ninguna reunión",
                                        "emoji": True
                                    },
                                    "value": "value-2"
                                }
                            ],
                            "action_id": "radio_buttons-action_t6"
                        }
            },
            {
                "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": emoji.emojize(":white_check_mark:") + " Si tuviste reuniones durante tu jornada, ¿pudiste despejar dudas sobre las tareas desarrolladas? " + emoji.emojize(" :thinking_face:")
                        },
                "accessory": {
                            "type": "radio_buttons",
                            "options": [
                                {
                                    "text": {
                                        "type": "plain_text",
                                        "text": "Si",
                                        "emoji": True
                                    },
                                    "value": "value-0"
                                },
                                {
                                    "text": {
                                        "type": "plain_text",
                                        "text": "No",
                                        "emoji": True
                                    },
                                    "value": "value-1"
                                },
                                {
                                    "text": {
                                        "type": "plain_text",
                                        "text": "No tuve ninguna reunion",
                                        "emoji": True
                                    },
                                    "value": "value-2"
                                },
                                {
                                    "text": {
                                        "type": "plain_text",
                                        "text": "Tuve reuniones, pero, no tenía dudas por aclarar",
                                        "emoji": True
                                    },
                                    "value": "value-3"
                                }
                            ],
                            "action_id": "radio_buttons-action_t7"
                        }
            },
            {
                "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": emoji.emojize(":white_check_mark:") + " En una escala de 1 a 5, como evalúa su productividad del día de hoy. " + emoji.emojize(":arrow_up:")
                        },
                "accessory": {
                            "type": "static_select",
                            "placeholder": {
                                "type": "plain_text",
                                "text": "seleccione un valor",
                                        "emoji": True
                            },
                            "options": [
                                {
                                    "text": {
                                        "type": "plain_text",
                                        "text": "1",
                                        "emoji": True
                                    },
                                    "value": "value-0"
                                },
                                {
                                    "text": {
                                        "type": "plain_text",
                                        "text": "2",
                                        "emoji": True
                                    },
                                    "value": "value-1"
                                },
                                {
                                    "text": {
                                        "type": "plain_text",
                                        "text": "3",
                                        "emoji": True
                                    },
                                    "value": "value-2"
                                },
                                {
                                    "text": {
                                        "type": "plain_text",
                                        "text": "4",
                                        "emoji": True
                                    },
                                    "value": "value-3"
                                },
                                {
                                    "text": {
                                        "type": "plain_text",
                                        "text": "5",
                                        "emoji": True
                                    },
                                    "value": "value-4"
                                }
                            ],
                            "action_id": "static_select-action_t8"
                        }
            },
            {
                "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": emoji.emojize(":white_check_mark:") + " ¿Cómo describirías la cantidad y distribución de tus descansos hoy? " + emoji.emojize(":coffee:")
                        },
                "accessory": {
                            "type": "static_select",
                            "placeholder": {
                                "type": "plain_text",
                                "text": "seleccione una respuesta",
                                        "emoji": True
                            },
                            "options": [
                                {
                                    "text": {
                                        "type": "plain_text",
                                        "text": "Suficientes y bien distribuidos",
                                        "emoji": True
                                    },
                                    "value": "value-0"
                                },
                                {
                                    "text": {
                                        "type": "plain_text",
                                        "text": "Suficientes pero mal distribuidos",
                                        "emoji": True
                                    },
                                    "value": "value-1"
                                },
                                {
                                    "text": {
                                        "type": "plain_text",
                                        "text": "Insuficientes pero bien distribuidos",
                                        "emoji": True
                                    },
                                    "value": "value-2"
                                },
                                {
                                    "text": {
                                        "type": "plain_text",
                                        "text": "Insuficientes y mal distribuidos",
                                        "emoji": True
                                    },
                                    "value": "value-3"
                                }
                            ],
                            "action_id": "static_select-action_t9"
                        }
            },
            {
                "type": "actions",
                        "elements": [
                            {
                                "type": "button",
                                "text": {
                                    "type": "plain_text",
                                    "text": "Enviar respuestas",
                                    "emoji": True
                                },
                                "value": "click_me_123",
                                "action_id": "actionId-0",
                                "style": "danger"
                            }
                        ]
            }
        ]
    }
    try:
        response = client.chat_postMessage(
            channel=user_id, text='formulario', blocks=formulario["blocks"])
        print("Formulario enviado con éxito - funcion_t")
    except SlackApiError as e:
        print(f"Error al enviar el formulario: {e.response['error']}")


# Definición de preguntas formulario de Inicio de jornada con botones y menús desplegables
def enviar_formulario_mm(user_id):
    jornada = {
        "blocks": [
            {
                "type": "header",
                        "text": {
                            "type": "plain_text",
                            "text": emoji.emojize(":robot_face:") + " Formulario de inicio de jornada " + emoji.emojize(":robot_face:"),
                            "emoji": True
                        }
            },
            {
                "type": "section",
                        "text": {
                            "type": "plain_text",
                            "text": "Responda las siguientes preguntas, esta información me será útil para ayudarte a optimizar tus tiempos y no te tomará mucho tiempo.",
                            "emoji": True
                        }
            },
            {
                "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": emoji.emojize(":white_check_mark:") + " ¿Tienes una agenda establecida para el desarrollo de tus actividades? " + emoji.emojize(":notebook:")
                        },
                "accessory": {
                            "type": "radio_buttons",
                            "options": [
                                    {
                                        "text": {
                                            "type": "plain_text",
                                            "text": "Si",
                                            "emoji": True
                                        },
                                        "value": "value-0"
                                    },
                                {
                                        "text": {
                                            "type": "plain_text",
                                            "text": "No",
                                            "emoji": True
                                        },
                                        "value": "value-1"
                                    }
                            ],
                            "action_id": "radio_buttons-action_m1"
                        }
            },
            {
                "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": emoji.emojize(":white_check_mark:") + " ¿Tienes reuniones agendadas? " + emoji.emojize(":calendar:")
                        },
                "accessory": {
                            "type": "radio_buttons",
                            "options": [
                                    {
                                        "text": {
                                            "type": "plain_text",
                                            "text": "Si",
                                            "emoji": True
                                        },
                                        "value": "value-0"
                                    },
                                {
                                        "text": {
                                            "type": "plain_text",
                                            "text": "No",
                                            "emoji": True
                                        },
                                        "value": "value-1"
                                    }
                            ],
                            "action_id": "radio_buttons-action_m2"
                        }
            },
            {
                "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": emoji.emojize(":white_check_mark:") + " En el caso de tener reuniones agendadas, indique con qué área/s te vas a reunir " + emoji.emojize(":file_folder:")
                        },
                "accessory": {
                            "type": "multi_static_select",
                            "placeholder": {
                                    "type": "plain_text",
                                    "text": "seleccione las respuestas",
                                    "emoji": True
                            },
                            "options": [
                                {
                                    "text": {
                                        "type": "plain_text",
                                        "text": "Data",
                                        "emoji": True
                                    },
                                    "value": "value-0"
                                },
                                {
                                    "text": {
                                        "type": "plain_text",
                                        "text": "Producto",
                                        "emoji": True
                                    },
                                    "value": "value-1"
                                },
                                {
                                    "text": {
                                        "type": "plain_text",
                                        "text": "Tecnología",
                                        "emoji": True
                                    },
                                    "value": "value-2"
                                },
                                {
                                    "text": {
                                        "type": "plain_text",
                                        "text": "Legal",
                                        "emoji": True
                                    },
                                    "value": "value-3"
                                },
                                {
                                    "text": {
                                        "type": "plain_text",
                                        "text": "Recursos humanos",
                                        "emoji": True
                                    },
                                    "value": "value-4"
                                },
                                {
                                    "text": {
                                        "type": "plain_text",
                                        "text": "Finanzas",
                                        "emoji": True
                                    },
                                    "value": "value-5"
                                },
                                {
                                    "text": {
                                        "type": "plain_text",
                                        "text": "Otro",
                                        "emoji": True
                                    },
                                    "value": "value-6"
                                }
                            ],
                            "action_id": "multi_static_select-action_m3"
                        }
            },
            {
                "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": emoji.emojize(":white_check_mark:") + " ¿Realizaste ayuno? " + emoji.emojize(":fork_and_knife:")
                        },
                "accessory": {
                            "type": "radio_buttons",
                            "options": [
                                    {
                                        "text": {
                                            "type": "plain_text",
                                            "text": "Si",
                                            "emoji": True
                                        },
                                        "value": "value-0"
                                    },
                                {
                                        "text": {
                                            "type": "plain_text",
                                            "text": "No",
                                            "emoji": True
                                        },
                                        "value": "value-1"
                                    }
                            ],
                            "action_id": "radio_buttons-action_m4"
                        }
            },
            {
                "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": emoji.emojize(":white_check_mark:") + " ¿Las tareas a desarrollar tienen un deadline establecido para hoy? " + emoji.emojize(":hourglass:")
                        },
                "accessory": {
                            "type": "radio_buttons",
                            "options": [
                                    {
                                        "text": {
                                            "type": "plain_text",
                                            "text": "Si",
                                            "emoji": True
                                        },
                                        "value": "value-0"
                                    },
                                {
                                        "text": {
                                            "type": "plain_text",
                                            "text": "No",
                                            "emoji": True
                                        },
                                        "value": "value-1"
                                    }
                            ],
                            "action_id": "radio_buttons-action_m5"
                        }
            },
            {
                "type": "actions",
                        "elements": [
                            {
                                "type": "button",
                                "text": {
                                    "type": "plain_text",
                                    "text": "Enviar respuestas",
                                    "emoji": True
                                },
                                "value": "click_me_123",
                                "action_id": "actionId-1",
                                "style": "danger"
                            }
                        ]
            }
        ]
    }
    try:
        response = client.chat_postMessage(
            channel=user_id, text="jornada", blocks=jornada["blocks"])
        print("Formulario enviado con éxito.")
    except SlackApiError as e:
        print(f"Error al enviar el formulario: {e.response['error']}")


###################### PROCESAMIENTO DE INTERACCIONES DEL USUARIO ###################
# Variable para almacenar diccionario con información del usuario (provisionalmente)
response_data = {}

# Función para el procesamiento de las cargas útiles resultado de la interacción del usuario


@app.route('/slack/interactions', methods=['POST'])
def handle_interactions():
    payload = json.loads(request.form.get('payload'))
    actions = payload['actions']

    # Se definen las variables a partir de la info del payload
    user_id = payload['user']['id']
    slack_username = payload['user']['username']
    full_name = payload['user']['name']

    # Inicializar el diccionario response_data con los valores previamente almacenados
    answers = response_data.copy()

    # Se almacenan los datos con formato clave - valor, pregunta: respuesta
    for action in actions:
        # print(action)
        if action['action_id'] == 'radio_buttons-action_t1':
            answer = action['selected_option']['text']['text']
            answers['alguna_tarea_te_llevo_mas_tiempo'] = answer
        elif action['action_id'] == 'radio_buttons-action_t2':
            answer = action['selected_option']['text']['text']
            answers['hay_tareas_que_se_pueden_automatizar'] = answer
        elif action['action_id'] == 'radio_buttons-action_t3':
            answer = action['selected_option']['text']['text']
            answers['tuviste_que_ir_a_algun_lugar_para_hacer_tus_tareas'] = answer
        elif action['action_id'] == 'radio_buttons-action_t4':
            answer = action['selected_option']['text']['text']
            answers['usaste_alguna_metodologia_para_optimizar_el_tiempo'] = answer
        elif action['action_id'] == 'radio_buttons-action_t5':
            answer = action['selected_option']['text']['text']
            answers['tuviste_reuniones_planificadas'] = answer
        elif action['action_id'] == 'radio_buttons-action_t6':
            answer = action['selected_option']['text']['text']
            answers['fueron_satisfactorias_las_reuniones'] = answer
        elif action['action_id'] == 'radio_buttons-action_t7':
            answer = action['selected_option']['text']['text']
            answers['pudiste_resolver_tus_dudas_sobre_el_trabajo'] = answer
        elif action['action_id'] == 'static_select-action_t8':
            answer = action['selected_option']['text']['text']
            answers['productividad_hoy'] = answer
        elif action['action_id'] == 'static_select-action_t9':
            answer = action['selected_option']['text']['text']
            answers['calificacion_descansos'] = answer
        # formulario de mañana
        elif action['action_id'] == 'radio_buttons-action_m1':
            answer = action['selected_option']['text']['text']
            answers['tienes_agenda_planeada'] = answer
        elif action['action_id'] == 'radio_buttons-action_m2':
            answer = action['selected_option']['text']['text']
            answers['tienes_reuniones_planeadas'] = answer
        elif action['action_id'] == 'multi_static_select-action_m3':
            areas = []
            for area in range(0, int(len(action['selected_options']))):
                areas.append(action['selected_options'][area]['text']['text'])
            answer = areas
            # print(answer)
            answers['con_que_areas_te_vas_a_reunir'] = answer
        elif action['action_id'] == 'radio_buttons-action_m4':
            answer = action['selected_option']['text']['text']
            answers['ayunaste_hoy'] = answer
        elif action['action_id'] == 'radio_buttons-action_m5':
            answer = action['selected_option']['text']['text']
            answers['tienes_deadline_hoy'] = answer

    # Actualizar el diccionario response_data con los nuevos valores almacenados
    response_data.update(answers)

    current_date = datetime.date.today()
    start_of_day = datetime.datetime.combine(
        current_date, datetime.datetime.min.time())
    end_of_day = start_of_day + timedelta(days=1)

    # Insertar las respuestas en la base de datos de MongoDB
    for action in actions:
        if action['action_id'] == 'actionId-0' and action['value'] == 'click_me_123':
            query = {"fecha_hora": {"$gte": start_of_day,
                                    "$lt": end_of_day}, "id_de_slack": user_id}
            results_t = final_questions_collection.find(query)
            contador_t = 0

            for t in results_t:
                contador_t += 1

            if contador_t == 0:
                response = emoji.emojize(":party_popper:") + emoji.emojize(":rocket:") + " ¡Respuestas recibidas! Gracias por completar el formulario. " + \
                    emoji.emojize(":rocket:") + emoji.emojize(":party_popper:")

                # Se verifican los valores a ingresar para segurar unicamente ingresar los valores del formulario correspondiente
                claves_tarde = ['alguna_tarea_te_llevo_mas_tiempo', 'hay_tareas_que_se_pueden_automatizar', 'tuviste_que_ir_a_algun_lugar_para_hacer_tus_tareas', 'usaste_alguna_metodologia_para_optimizar_el_tiempo',
                                'tuviste_reuniones_planificadas', 'fueron_satisfactorias_las_reuniones', 'pudiste_resolver_tus_dudas_sobre_el_trabajo', 'productividad_hoy', 'calificacion_descansos']
                respuestas_tarde = {clave: response_data.get(
                    clave, 'null') for clave in claves_tarde}

                # Insertar datos a mongo formulario tarde
                insert_final_questions(
                    user_id, slack_username, full_name, respuestas_tarde)

                # eliminar datos de la variable global 'response_data'
                response_data.clear()
                try:
                    client.chat_postMessage(channel=user_id, text=response)
                    print("Respuestas enviadas con éxito.")
                except SlackApiError as e:
                    print(
                        f"Error al enviar las respuestas: {e.response['error']}")
                break
            else:
                texto = emoji.emojize(
                    ":no_entry_sign:") + " Lo siento, al parecer ya has diligenciado el formulario el día de hoy. " + emoji.emojize(":no_entry_sign:")
                client.chat_postMessage(channel=user_id, text=texto)

        elif action['action_id'] == 'actionId-1' and action['value'] == 'click_me_123':
            query = {"fecha_hora": {"$gte": start_of_day,
                                    "$lt": end_of_day}, "id_de_slack": user_id}
            results_m = initial_questions_collection.find(query)
            contador_m = 0

            for m in results_m:
                contador_m += 1

            if contador_m == 0:
                response = emoji.emojize(":party_popper:") + emoji.emojize(":rocket:") + " ¡Respuestas recibidas! Gracias por completar el formulario. " + \
                    emoji.emojize(":rocket:") + emoji.emojize(":party_popper:")

                # Se verifican los valores a ingresar para segurar unicamente ingresar los valores del formulario correspondiente
                claves_mañana = ['tienes_agenda_planeada', 'tienes_reuniones_planeadas',
                                 'con_que_areas_te_vas_a_reunir', 'ayunaste_hoy', 'tienes_deadline_hoy']
                respuestas_mañana = {clave: response_data.get(
                    clave, 'null') for clave in claves_mañana}

                # Insertar datos a mongo formulario mañana
                print(response_data)
                insert_initial_questions(
                    user_id, slack_username, full_name, respuestas_mañana)
                # eliminar datos de la variable global 'response_data'
                response_data.clear()
                try:
                    client.chat_postMessage(channel=user_id, text=response)
                    print("Respuestas enviadas con éxito.")
                except SlackApiError as e:
                    print(
                        f"Error al enviar las respuestas: {e.response['error']}")
                break

            else:
                texto = emoji.emojize(
                    ":no_entry_sign:") + " Lo siento, al parecer ya has diligenciado el formulario el día de hoy. " + emoji.emojize(":no_entry_sign:")
                client.chat_postMessage(channel=user_id, text=texto)

    return jsonify({'response_action': 'clear'})

############################### PROGRAMACIÓN DE ENVIÓ DE FORMULARIO ########################


# Lista de usuarios con el holobot en holocruxe
response = client.users_list()

# Para enviar a todos los usuarios se debe habilitar users
# users = response["members"] #esta es la que se debe dejar para generalizacion
'''
users = {'id': 'U05A9P96PM1', 'team_id': 'T055APAMLKT', 'name': 'cpfedericogomez', 'deleted': False, 'color': 'db3150', 'real_name': 'Federico G�mez', 'tz': 'America/Buenos_Aires', 'tz_label': 'Argentina Time', 'tz_offset': -10800, 'profile': {'title': '', 'phone': '', 'skype': '', 'real_name': 'Federico G�mez', 'real_name_normalized': 'Federico Gomez', 'display_name': 'Federico G�mez', 'display_name_normalized': 'Federico Gomez', 'fields': None, 'status_text': '', 'status_emoji': '', 'status_emoji_display_info': [], 'status_expiration': 0, 'avatar_hash': 'dfc49efefc1b', 'image_original': 'https://avatars.slack-edge.com/2023-05-31/5365239212929_dfc49efefc1b20784fc5_original.png', 'is_custom_image': True, 'first_name': 'Federico', 'last_name': 'G�mez', 'image_24': 'https://avatars.slack-edge.com/2023-05-31/5365239212929_dfc49efefc1b20784fc5_24.png',
                                                                                                                                                                                                                                                    'image_32': 'https://avatars.slack-edge.com/2023-05-31/5365239212929_dfc49efefc1b20784fc5_32.png', 'image_48': 'https://avatars.slack-edge.com/2023-05-31/5365239212929_dfc49efefc1b20784fc5_48.png', 'image_72': 'https://avatars.slack-edge.com/2023-05-31/5365239212929_dfc49efefc1b20784fc5_72.png', 'image_192': 'https://avatars.slack-edge.com/2023-05-31/5365239212929_dfc49efefc1b20784fc5_192.png', 'image_512': 'https://avatars.slack-edge.com/2023-05-31/5365239212929_dfc49efefc1b20784fc5_512.png', 'image_1024': 'https://avatars.slack-edge.com/2023-05-31/5365239212929_dfc49efefc1b20784fc5_1024.png', 'status_text_canonical': '', 'team': 'T055APAMLKT'}, 'is_admin': False, 'is_owner': False, 'is_primary_owner': False, 'is_restricted': False, 'is_ultra_restricted': False, 'is_bot': False, 'is_app_user': False, 'updated': 1685560162, 'is_email_confirmed': True, 'who_can_share_contact_card': 'EVERYONE'}, {'id': 'U05ANG76U3B', 'team_id': 'T055APAMLKT', 'name': 'johan.gbc', 'deleted': False, 'color': '235e5b', 'real_name': 'Johan Andres Rodriguez Rojas', 'tz': 'America/Bogota', 'tz_label': 'South America Pacific Standard Time', 'tz_offset': -18000, 'profile': {'title': '', 'phone': '', 'skype': '', 'real_name': 'Johan Andres Rodriguez Rojas', 'real_name_normalized': 'Johan Andres Rodriguez Rojas', 'display_name': 'Johan Andres Rodriguez Rojas', 'display_name_normalized': 'Johan Andres Rodriguez Rojas', 'fields': None, 'status_text': '', 'status_emoji': '', 'status_emoji_display_info': [], 'status_expiration': 0, 'avatar_hash': '32defbe53b9e', 'image_original': 'https://avatars.slack-edge.com/2023-06-07/5382332660886_32defbe53b9e09a5aecd_original.jpg', 'is_custom_image': True, 'huddle_state': 'default_unset', 'huddle_state_expiration_ts': 0, 'first_name': 'Johan', 'last_name': 'Andres Rodriguez Rojas',
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  'image_24': 'https://avatars.slack-edge.com/2023-06-07/5382332660886_32defbe53b9e09a5aecd_24.jpg', 'image_32': 'https://avatars.slack-edge.com/2023-06-07/5382332660886_32defbe53b9e09a5aecd_32.jpg', 'image_48': 'https://avatars.slack-edge.com/2023-06-07/5382332660886_32defbe53b9e09a5aecd_48.jpg', 'image_72': 'https://avatars.slack-edge.com/2023-06-07/5382332660886_32defbe53b9e09a5aecd_72.jpg', 'image_192': 'https://avatars.slack-edge.com/2023-06-07/5382332660886_32defbe53b9e09a5aecd_192.jpg', 'image_512': 'https://avatars.slack-edge.com/2023-06-07/5382332660886_32defbe53b9e09a5aecd_512.jpg', 'image_1024': 'https://avatars.slack-edge.com/2023-06-07/5382332660886_32defbe53b9e09a5aecd_1024.jpg', 'status_text_canonical': '', 'team': 'T055APAMLKT'}, 'is_admin': True, 'is_owner': False, 'is_primary_owner': False, 'is_restricted': False, 'is_ultra_restricted': False, 'is_bot': False, 'is_app_user': False, 'updated': 1686241517, 'is_email_confirmed': True, 'who_can_share_contact_card': 'EVERYONE'}, {'id': 'U05ANG79VFT', 'team_id': 'T055APAMLKT', 'name': 'santiagocioffi', 'deleted': False, 'color': '9e3997', 'real_name': 'Santiago Cioffi', 'tz': 'America/Buenos_Aires', 'tz_label': 'Argentina Time', 'tz_offset': -10800, 'profile': {'title': '', 'phone': '', 'skype': '', 'real_name': 'Santiago Cioffi', 'real_name_normalized': 'Santiago Cioffi', 'display_name': 'Santiago Cioffi', 'display_name_normalized': 'Santiago Cioffi', 'fields': None, 'status_text': '', 'status_emoji': '', 'status_emoji_display_info': [], 'status_expiration': 0, 'avatar_hash': '51186a830bf4', 'image_original': 'https://avatars.slack-edge.com/2023-05-31/5343725390630_51186a830bf41ecdeb90_original.jpg', 'is_custom_image': True, 'first_name': 'Santiago', 'last_name': 'Cioffi', 'image_24': 'https://avatars.slack-edge.com/2023-05-31/5343725390630_51186a830bf41ecdeb90_24.jpg',
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           'image_32': 'https://avatars.slack-edge.com/2023-05-31/5343725390630_51186a830bf41ecdeb90_32.jpg', 'image_48': 'https://avatars.slack-edge.com/2023-05-31/5343725390630_51186a830bf41ecdeb90_48.jpg', 'image_72': 'https://avatars.slack-edge.com/2023-05-31/5343725390630_51186a830bf41ecdeb90_72.jpg', 'image_192': 'https://avatars.slack-edge.com/2023-05-31/5343725390630_51186a830bf41ecdeb90_192.jpg', 'image_512': 'https://avatars.slack-edge.com/2023-05-31/5343725390630_51186a830bf41ecdeb90_512.jpg', 'image_1024': 'https://avatars.slack-edge.com/2023-05-31/5343725390630_51186a830bf41ecdeb90_1024.jpg', 'status_text_canonical': '', 'team': 'T055APAMLKT'}, 'is_admin': False, 'is_owner': False, 'is_primary_owner': False, 'is_restricted': False, 'is_ultra_restricted': False, 'is_bot': False, 'is_app_user': False, 'updated': 1685541986, 'is_email_confirmed': True, 'who_can_share_contact_card': 'EVERYONE'}, {'id': 'U059VB53R9D', 'team_id': 'T055APAMLKT', 'name': 'agustinvaldes.central', 'deleted': False, 'color': '53b759', 'real_name': 'Agustin Valdes', 'tz': 'America/Buenos_Aires', 'tz_label': 'Argentina Time', 'tz_offset': -10800, 'profile': {'title': '', 'phone': '', 'skype': '', 'real_name': 'Agustin Valdes', 'real_name_normalized': 'Agustin Valdes', 'display_name': 'Agustin Valdes', 'display_name_normalized': 'Agustin Valdes', 'fields': None, 'status_text': '', 'status_emoji': '', 'status_emoji_display_info': [], 'status_expiration': 0, 'avatar_hash': '83c429bafd34', 'image_original': 'https://avatars.slack-edge.com/2023-05-31/5350223916387_83c429bafd347f72ee7e_original.png', 'is_custom_image': True, 'first_name': 'Agustin', 'last_name': 'Valdes', 'image_24': 'https://avatars.slack-edge.com/2023-05-31/5350223916387_83c429bafd347f72ee7e_24.png',
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        'image_32': 'https://avatars.slack-edge.com/2023-05-31/5350223916387_83c429bafd347f72ee7e_32.png', 'image_48': 'https://avatars.slack-edge.com/2023-05-31/5350223916387_83c429bafd347f72ee7e_48.png', 'image_72': 'https://avatars.slack-edge.com/2023-05-31/5350223916387_83c429bafd347f72ee7e_72.png', 'image_192': 'https://avatars.slack-edge.com/2023-05-31/5350223916387_83c429bafd347f72ee7e_192.png', 'image_512': 'https://avatars.slack-edge.com/2023-05-31/5350223916387_83c429bafd347f72ee7e_512.png', 'image_1024': 'https://avatars.slack-edge.com/2023-05-31/5350223916387_83c429bafd347f72ee7e_1024.png', 'status_text_canonical': '', 'team': 'T055APAMLKT'}, 'is_admin': False, 'is_owner': False, 'is_primary_owner': False, 'is_restricted': False, 'is_ultra_restricted': False, 'is_bot': False, 'is_app_user': False, 'updated': 1685541720, 'is_email_confirmed': True, 'who_can_share_contact_card': 'EVERYONE'}
'''
# usuarios recortados
users = {'id': 'U05A9P96PM1', 'team_id': 'T055APAMLKT', 'name': 'cpfedericogomez', 'deleted': False, 'color': 'db3150', 'real_name': 'Federico G�mez', 'tz': 'America/Buenos_Aires', 'tz_label': 'Argentina Time', 'tz_offset': -10800, 'profile': {'title': '', 'phone': '', 'skype': '', 'real_name': 'Federico G�mez', 'real_name_normalized': 'Federico Gomez', 'display_name': 'Federico G�mez', 'display_name_normalized': 'Federico Gomez', 'fields': None, 'status_text': '', 'status_emoji': '', 'status_emoji_display_info': [], 'status_expiration': 0, 'avatar_hash': 'dfc49efefc1b', 'image_original': 'https://avatars.slack-edge.com/2023-05-31/5365239212929_dfc49efefc1b20784fc5_original.png', 'is_custom_image': True, 'first_name': 'Federico', 'last_name': 'G�mez', 'image_24': 'https://avatars.slack-edge.com/2023-05-31/5365239212929_dfc49efefc1b20784fc5_24.png',
                                                                                                                                                                                                                                                    'image_32': 'https://avatars.slack-edge.com/2023-05-31/5365239212929_dfc49efefc1b20784fc5_32.png', 'image_48': 'https://avatars.slack-edge.com/2023-05-31/5365239212929_dfc49efefc1b20784fc5_48.png', 'image_72': 'https://avatars.slack-edge.com/2023-05-31/5365239212929_dfc49efefc1b20784fc5_72.png', 'image_192': 'https://avatars.slack-edge.com/2023-05-31/5365239212929_dfc49efefc1b20784fc5_192.png', 'image_512': 'https://avatars.slack-edge.com/2023-05-31/5365239212929_dfc49efefc1b20784fc5_512.png', 'image_1024': 'https://avatars.slack-edge.com/2023-05-31/5365239212929_dfc49efefc1b20784fc5_1024.png', 'status_text_canonical': '', 'team': 'T055APAMLKT'}, 'is_admin': False, 'is_owner': False, 'is_primary_owner': False, 'is_restricted': False, 'is_ultra_restricted': False, 'is_bot': False, 'is_app_user': False, 'updated': 1685560162, 'is_email_confirmed': True, 'who_can_share_contact_card': 'EVERYONE'}, {'id': 'U05ANG76U3B', 'team_id': 'T055APAMLKT', 'name': 'johan.gbc', 'deleted': False, 'color': '235e5b', 'real_name': 'Johan Andres Rodriguez Rojas', 'tz': 'America/Bogota', 'tz_label': 'South America Pacific Standard Time', 'tz_offset': -18000, 'profile': {'title': '', 'phone': '', 'skype': '', 'real_name': 'Johan Andres Rodriguez Rojas', 'real_name_normalized': 'Johan Andres Rodriguez Rojas', 'display_name': 'Johan Andres Rodriguez Rojas', 'display_name_normalized': 'Johan Andres Rodriguez Rojas', 'fields': None, 'status_text': '', 'status_emoji': '', 'status_emoji_display_info': [], 'status_expiration': 0, 'avatar_hash': '32defbe53b9e', 'image_original': 'https://avatars.slack-edge.com/2023-06-07/5382332660886_32defbe53b9e09a5aecd_original.jpg', 'is_custom_image': True, 'huddle_state': 'default_unset', 'huddle_state_expiration_ts': 0, 'first_name': 'Johan', 'last_name': 'Andres Rodriguez Rojas',
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  'image_24': 'https://avatars.slack-edge.com/2023-06-07/5382332660886_32defbe53b9e09a5aecd_24.jpg', 'image_32': 'https://avatars.slack-edge.com/2023-06-07/5382332660886_32defbe53b9e09a5aecd_32.jpg', 'image_48': 'https://avatars.slack-edge.com/2023-06-07/5382332660886_32defbe53b9e09a5aecd_48.jpg', 'image_72': 'https://avatars.slack-edge.com/2023-06-07/5382332660886_32defbe53b9e09a5aecd_72.jpg', 'image_192': 'https://avatars.slack-edge.com/2023-06-07/5382332660886_32defbe53b9e09a5aecd_192.jpg', 'image_512': 'https://avatars.slack-edge.com/2023-06-07/5382332660886_32defbe53b9e09a5aecd_512.jpg', 'image_1024': 'https://avatars.slack-edge.com/2023-06-07/5382332660886_32defbe53b9e09a5aecd_1024.jpg', 'status_text_canonical': '', 'team': 'T055APAMLKT'}, 'is_admin': True, 'is_owner': False, 'is_primary_owner': False, 'is_restricted': False, 'is_ultra_restricted': False, 'is_bot': False, 'is_app_user': False, 'updated': 1686241517, 'is_email_confirmed': True, 'who_can_share_contact_card': 'EVERYONE'}


# id de usuario de prueba para hacer controles
usuario_id = 'U05ANG76U3B'


################################### INTEGRACIÓN CON EL MODELO DE APRENDIZAJE  ############################

# Ruta del modelo
export_path = 'modelo_guardado/1687453240'

# Cargar el modelo exportado
loaded_model = tf.saved_model.load(export_path)

# funcion de predicciones


def predict_new_instance(instance, model):
    # Convertir la instancia a un diccionario
    instance_dict = {col: [instance[col]] for col in instance.keys()}

    # Crear un Dataset a partir del diccionario
    dataset = tf.data.Dataset.from_tensor_slices(instance_dict)

    # Agrupar el Dataset
    dataset = dataset.batch(1)

    # Serializar la instancia
    serialized_instance = tf.train.Example(features=tf.train.Features(feature={
        col: tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(str(instance[col]))])) for col in instance.keys()
    })).SerializeToString()

    # Realizar predicciones con el modelo cargado
    predictions = model.signatures['serving_default'](
        tf.constant([serialized_instance]))

    # Obtener los resultados de predicción
    predicted_values = predictions['outputs'].numpy()

    # Redondear la predicción a un número entero
    predicted_productivity = round(predicted_values[0][0])

    # Crear un mensaje basado en la predicción
    if predicted_productivity == instance['productividad_hoy']:
        if instance['productividad_hoy'] == 1:
            message = f"{emoji.emojize(':weary:')} Hoy fue un día difícil, pero mañana será mejor! {emoji.emojize(':rocket:')}\nTu respuesta coincidió con la productividad del modelo. {emoji.emojize(':rocket:')}\nProductividad predicha: {predicted_productivity}\nProductividad real: {instance['productividad_hoy']}"
        elif instance['productividad_hoy'] == 2:
            message = f"{emoji.emojize(':neutral_face:')} No fue tu mejor día, pero sé que puedes mejorar! {emoji.emojize(':rocket:')}\nTu respuesta coincidió con la productividad del modelo. {emoji.emojize(':rocket:')}\nProductividad predicha: {predicted_productivity}\nProductividad real: {instance['productividad_hoy']}"
        elif instance['productividad_hoy'] == 3:
            message = f"{emoji.emojize(':slightly_smiling_face:')} Hiciste un buen trabajo hoy, pero sé que puedes hacerlo aún mejor! {emoji.emojize(':rocket:')}\nTu respuesta coincidió con la productividad del modelo. {emoji.emojize(':rocket:')}\nProductividad predicha: {predicted_productivity}\nProductividad real: {instance['productividad_hoy']}"
        elif instance['productividad_hoy'] == 4:
            message = f"{emoji.emojize(':smiley:')} ¡Excelente trabajo hoy! ¡Sigue así!{emoji.emojize(':rocket:')}\nTu respuesta coincidió con la productividad del modelo. {emoji.emojize(':rocket:')}\nProductividad predicha: {predicted_productivity}\nProductividad real: {instance['productividad_hoy']}"
        elif instance['productividad_hoy'] == 5:
            message = f"{emoji.emojize(':star-struck:')} ¡Hiciste un trabajo increíble hoy! ¡Sigue superándote!{emoji.emojize(':rocket:')}\nTu respuesta coincidió con la productividad del modelo. {emoji.emojize(':rocket:')}\nProductividad predicha: {predicted_productivity}\nProductividad real: {instance['productividad_hoy']}"
    elif predicted_productivity < instance['productividad_hoy']:
        if instance['productividad_hoy'] == 5:
            message = f"{emoji.emojize(':star-struck:')} ¡Hiciste un trabajo increíble hoy! ¡Sigue superándote!{emoji.emojize(':rocket:')}\nTu respuesta superó la productividad del modelo.{emoji.emojize(':rocket:')}\nProductividad predicha: {predicted_productivity}\nProductividad real: {instance['productividad_hoy']}"
        elif predicted_productivity == 4:
            message = f"{emoji.emojize(':slightly_smiling_face:')} Parece que tu productividad fue menor a lo esperado, pero sigue estando muy bien!{emoji.emojize(':rocket:')}\n¡Sigue así!{emoji.emojize(':rocket:')}\nProductividad predicha: {predicted_productivity}\nProductividad real: {instance['productividad_hoy']}"
        elif predicted_productivity <= 3:
            message = f"{emoji.emojize(':slightly_frowning_face:')} Tu productividad fue un poco menor a lo esperado.\n¡Mañana es otro día para mejorar!{emoji.emojize(':rocket:')}\nProductividad predicha: {predicted_productivity}\nProductividad real: {instance['productividad_hoy']}"
        elif predicted_productivity <= 2:
            message = f"{emoji.emojize(':disappointed:')} Tu productividad fue un poco menor a lo esperado, pero mañana será mejor!{emoji.emojize(':rocket:')}\nProductividad predicha: {predicted_productivity}\nProductividad real: {instance['productividad_hoy']}"
    elif predicted_productivity > instance['productividad_hoy']:
        if predicted_productivity == 5:
            message = f"{emoji.emojize(':fire:')} ¡Superaste las expectativas hoy!{emoji.emojize(':rocket:')}\n¡Excelente trabajo!{emoji.emojize(':rocket:')}\nProductividad predicha: {predicted_productivity}\nProductividad real: {instance['productividad_hoy']}"
        elif predicted_productivity == 4:
            message = f"{emoji.emojize(':smiley:')} ¡Hiciste un buen trabajo hoy!{emoji.emojize(':rocket:')}\n¡Sigue así!{emoji.emojize(':rocket:')}\nProductividad predicha: {predicted_productivity}\nProductividad real: {instance['productividad_hoy']}"
        elif predicted_productivity == 3:
            message = f"{emoji.emojize(':slightly_smiling_face:')} Hiciste un buen trabajo hoy, pero sé que puedes hacerlo aún mejor!{emoji.emojize(':rocket:')}\nProductividad predicha: {predicted_productivity}\nProductividad real: {instance['productividad_hoy']}"
        elif predicted_productivity == 2:
            message = f"{emoji.emojize(':neutral_face:')} No fue tu mejor día, pero sé que puedes mejorar!{emoji.emojize(':rocket:')}\nProductividad predicha: {predicted_productivity}\nProductividad real: {instance['productividad_hoy']}"
        elif predicted_productivity == 1:
            message = f"{emoji.emojize(':weary:')} Hoy fue un día difícil, pero mañana será mejor!{emoji.emojize(':rocket:')}\nProductividad predicha: {predicted_productivity}\nProductividad real: {instance['productividad_hoy']}"

    return message

########################### HORAS DE ENVIO DEL FORMULARIO ########################


# Formulario de inicio de jornada
hora_form_mm = '08:23'
# Formulario de finalizacion de jornada
hora_form_tt = '08:24'

# Variable global para el manejo del servidor dentro de la funcion
stop_schedule = False

########################### RESPUESTA DEL MODELO AL USUARIO #######################


def respuesta_modelo(user_id):
    try:
        # Extracción de datos para alimentar el modelo por usuario
        current_date = datetime.date.today()
        start_of_day = datetime.datetime.combine(
            current_date, datetime.datetime.min.time())
        end_of_day = start_of_day + timedelta(days=1)

        query = {"fecha_hora": {"$gte": start_of_day,
                                "$lt": end_of_day}, "id_de_slack": user_id}
        results_m = initial_questions_collection.find(query)
        results_t = final_questions_collection.find(query)

        contador_m = 0
        contador_t = 0
        for m in results_m:
            contador_m += 1

        for t in results_t:
            contador_t += 1

        if contador_m == 1 and contador_t == 1:
            results_m = initial_questions_collection.find(query)
            results_t = final_questions_collection.find(query)

            final = {}
            for result in results_m:
                # print(len(result))
                resultado = {
                    'id_de_slack': result['id_de_slack'],
                    'tienes_agenda_planeada': result['tienes_agenda_planeada'],
                    'tienes_reuniones_planeadas': result['tienes_reuniones_planeadas'],
                    'ayunaste_hoy': result['ayunaste_hoy'],
                    'con_que_areas_te_vas_a_reunir': result['con_que_areas_te_vas_a_reunir'],
                    'tienes_deadline_hoy': result['tienes_deadline_hoy']
                }
                final.update(resultado)

            for result in results_t:
                resultado = {
                    'alguna_tarea_te_llevo_mas_tiempo': result['alguna_tarea_te_llevo_mas_tiempo'],
                    'hay_tareas_que_se_pueden_automatizar': result['hay_tareas_que_se_pueden_automatizar'],
                    'tuviste_que_ir_a_algun_lugar_para_hacer_tus_tareas': result['tuviste_que_ir_a_algun_lugar_para_hacer_tus_tareas'],
                    'usaste_alguna_metodologia_para_optimizar_el_tiempo': result['usaste_alguna_metodologia_para_optimizar_el_tiempo'],
                    'fueron_satisfactorias_las_reuniones': result['fueron_satisfactorias_las_reuniones'],
                    'pudiste_resolver_tus_dudas_sobre_el_trabajo': result['pudiste_resolver_tus_dudas_sobre_el_trabajo'],
                    'tuviste_reuniones_planificadas': result['tuviste_reuniones_planificadas'],
                    'productividad_hoy': int(result['productividad_hoy']),
                    'calificacion_descansos': result['calificacion_descansos']
                }
                final.update(resultado)

            message = predict_new_instance(final, loaded_model)
            client.chat_postMessage(
                channel=user_id, text=message)

        else:
            message = 'Lo siento, no tiene los registros diarios suficientes para generar una predicción'
            client.chat_postMessage(
                channel=user_id, text=message)

    except Exception as e:
        print("Ocurrió un error:", str(e))

########################### BUCLE PRINCIPAL DEL BOT ########################
# Función principal encargada de enviar los formularios, recolectar las respuestas y entregar predicciones a los usuarios.


def run_schedule():
    # Función para manejar la señal de cierre del servidor
    print('aqui toy')

    def handle_exit(signum, frame):
        global stop_schedule
        stop_schedule = True
        print("Señal de cierre recibida. Deteniendo el programa...")

    # Manejar la señal SIGINT a partir de la funcion 'handle_exit'
    signal.signal(signal.SIGINT, handle_exit)

    # Ciclo para alternar la verificacion de horarios y la ejecucion de formularios con interaccion
    # Se ejecuta cunado 'stop_schedule' sea True
    while not stop_schedule:
        try:
            current_time = datetime.datetime.now().strftime("%H:%M")
            current_day = datetime.datetime.now().strftime("%A")

            # Verificacion de los dias de ejecicucion del codigo
            if current_day != "Saturday" and current_day != "Sunday":
                # if current_day == "Saturday" or current_day == "Sunday":

                # Lógica para enviar el formulario MM y activar el servidor
                if current_time == hora_form_mm:
                    try:
                        for user in users:
                            try:
                                user_id = user["id"]
                                enviar_formulario_mm(user_id)
                                print(
                                    'Formulario enviado con exito al usuario', user_id)
                            except Exception as e:
                                print("Error al enviar formulario:", str(e))
                    except Exception as e:
                        print("Error al enviar otro formulario:", str(e))

                    print("Formulario mañana enviado con éxito.")

                    # Iniciar el servidor Flask para recibir respuestas de los usuarios
                    server = make_server('localhost', 8083, app)
                    server_thread = threading.Thread(
                        target=server.serve_forever)
                    server_thread.start()

                    # Establecer un temporizador para detener el servidor después de x tiempo
                    time.sleep(60)  # en este caso un minuto

                    # Detener el servidor
                    server.shutdown()

                    schedule.run_pending()

                if current_time == hora_form_tt:
                    # Lógica para enviar el otro formulario
                    try:
                        for user in users:
                            try:
                                user_id = user["id"]
                                enviar_formulario_tt(user_id)
                                print(
                                    'Formulario enviado con exito al usuario', user_id)
                            except Exception as e:
                                print("Error al enviar formulario:", str(e))

                    except Exception as e:
                        print("Error al enviar otro formulario:", str(e))

                    print("Formulario tarde enviado con éxito.")

                    server = make_server('localhost', 8083, app)
                    server_thread = threading.Thread(
                        target=server.serve_forever)
                    server_thread.start()

                    # Establecer un temporizador para detener el servidor después de 1 hora
                    time.sleep(60)

                    # Extraccion de datos para alimentar el modelo por usuario

                    for user in users:
                        try:
                            user_id = user["id"]
                            respuesta_modelo(user_id)
                            print(
                                'Respuesta del modelo enviada con exito', user_id)
                        except Exception as e:
                            print("Error al enviar formulario:", str(e))

                    # Detener el servidor después de x tiempo
                    server.shutdown()

                    # break  # Salir del bucle después de enviar el formulario y detener el servidor

                schedule.run_pending()
                print("Verificando tareas programadas...2", current_time)
                time.sleep(20)
        except Exception as e:
            print("Error en la ejecución del programa:", str(e))


if __name__ == "__main__":
    run_schedule()
    app.run(debug=True, port=8083)
