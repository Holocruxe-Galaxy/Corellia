import prediction
import pandas as pd
nueva_rta = {'_id': {249: '64c3cbd6622d8ae354306271'},
 'fecha_hora': {249: '2023-09-15 11:08:22.899000'},
 'id_de_slack': {249: 'slack_user2'},
 'nombre_usuario_slack': {249: 'slack_username3'},
 'nombre_completo': {249: 'Mamen Álvaro Velázquez'},
 'tienes_agenda_planeada': {249: 'Si'},
 'tienes_reuniones_planeadas': {249: 'No'},
 'con_que_areas_te_vas_a_reunir': {249: 'Otro'},
 'ayunaste_hoy': {249: 'No'},
 'tienes_deadline_hoy': {249: 'Si'},
 'alguna_tarea_te_llevo_mas_tiempo': {249: 'Si'},
 'hay_tareas_que_se_pueden_automatizar': {249: 'Si'},
 'tuviste_que_ir_a_algun_lugar_para_hacer_tus_tareas': {249: 'No'},
 'usaste_alguna_metodologia_para_optimizar_el_tiempo': {249: 'Si'},
 'fueron_satisfactorias_las_reuniones': {249: 'No'},
 'pudiste_resolver_tus_dudas_sobre_el_trabajo': {249: 'No'},
 'tuviste_reuniones_planificadas': {249: 'Si'},
 'productividad_hoy': {249: 2},
 'calificacion_descansos': {249: 'Insuficientes pero bien distribuidos'}}
nueva_rta_df = pd.DataFrame(nueva_rta)
nueva_pred = prediction.predict(nueva_rta_df)
print("La prediccion predicha es:", nueva_pred)