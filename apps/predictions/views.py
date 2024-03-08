from django.shortcuts import render
import pandas as pd
import os
from django.conf import settings
import pickle
from django.http import FileResponse
from django.views import View
import os
import mimetypes


''' Funcion para modelo NB '''
def predictions_nb(request):
    path_data = os.path.join(settings.BASE_DIR, 'apps/predictions/data_set/card_transdata.csv')
    path_model = os.path.join(settings.BASE_DIR, 'apps/predictions/models/NBModel.pkl')

    # Leer el archivo CSV
    df = pd.read_csv(path_data)


    # Cargar el modelo
    with open(path_model, 'rb') as model_file:
        nb_model = pickle.load(model_file)

    # Número total de registros en el DataFrame
    total_registros = len(df)

    # Tamaño del lote (batch)
    tamano_lote = 1000

    # Lista para almacenar los resultados de cada lote
    resultados = []

    # Iterar sobre lotes de 1000 registros
    for inicio in range(0, total_registros, tamano_lote):
        fin = min(inicio + tamano_lote, total_registros)
        
        # Tomar el lote actual
        lote_df = df.iloc[inicio:fin]

        # Asegurarse de tener las columnas necesarias para las predicciones
        # Puedes ajustar esto según las características utilizadas por tu modelo
        if 'fraud' in lote_df.columns:
            X_lote = lote_df.drop('fraud', axis=1)
        else:
            X_lote = lote_df.copy()

        # Hacer predicciones
        predictions = nb_model.predict(X_lote)

        # Agregar las predicciones al DataFrame del lote
        lote_df['predicted_fraud'] = predictions

        # Agregar el lote a la lista de resultados
        resultados.append(lote_df)

    # Concatenar todos los lotes en un solo DataFrame
    df_resultados = pd.concat(resultados, ignore_index=True)

    # Obtener solo las primeras 100 filas del DataFrame de resultados
    df_resultados_preview = df_resultados.head(100)

    # Guardar el DataFrame de resultados en un nuevo archivo CSV (opcional)
    df_resultados_preview.to_csv('apps/predictions/data_set/predicciones_fraude_resultados_preview.csv', index=False)

    # Guardar el DataFrame de resultados en un nuevo archivo CSV
    df_resultados.to_csv('apps/predictions/data_set/predicciones_fraude_resultados_nb.csv', index=False)
    
    # Renderizar un template con información sobre el procesamiento
    return render(request, 'predictions_nb.html', {'archivo_resultados': 'apps/predictions/data_set/predicciones_fraude_resultados_nb.csv', 'df_predicciones': df_resultados_preview})


''' Funcion para el modelo Gboost '''
def predictions_gboost(request):
    path_data = os.path.join(settings.BASE_DIR, 'apps/predictions/data_set/card_transdata.csv')
    path_model = os.path.join(settings.BASE_DIR, 'apps/predictions/models/Gboost_model.pkl')

    # Leer el archivo CSV
    df = pd.read_csv(path_data)


    # Cargar el modelo
    with open(path_model, 'rb') as model_file:
        gb_model = pickle.load(model_file)

    # Número total de registros en el DataFrame
    total_registros = len(df)

    # Tamaño del lote (batch)
    tamano_lote = 1000

    # Lista para almacenar los resultados de cada lote
    resultados = []

    # Iterar sobre lotes de 1000 registros
    for inicio in range(0, total_registros, tamano_lote):
        fin = min(inicio + tamano_lote, total_registros)
        
        # Tomar el lote actual
        lote_df = df.iloc[inicio:fin]

        # Asegurarse de tener las columnas necesarias para las predicciones
        # Puedes ajustar esto según las características utilizadas por tu modelo
        if 'fraud' in lote_df.columns:
            X_lote = lote_df.drop('fraud', axis=1)
        else:
            X_lote = lote_df.copy()

        # Hacer predicciones
        predictions = gb_model.predict(X_lote)

        # Agregar las predicciones al DataFrame del lote
        lote_df['predicted_fraud'] = predictions

        # Agregar el lote a la lista de resultados
        resultados.append(lote_df)

    # Concatenar todos los lotes en un solo DataFrame
    df_resultados = pd.concat(resultados, ignore_index=True)

    # Obtener solo las primeras 100 filas del DataFrame de resultados
    df_resultados_preview = df_resultados.head(100)

    # Guardar el DataFrame de resultados en un nuevo archivo CSV (opcional)
    df_resultados_preview.to_csv('apps/predictions/data_set/predicciones_fraude_resultados_preview.csv', index=False)

    # Guardar el DataFrame de resultados en un nuevo archivo CSV
    df_resultados.to_csv('apps/predictions/data_set/predicciones_fraude_resultados_gb.csv', index=False)

    # Renderizar un template con información sobre el procesamiento
    return render(request, 'predictions_gboost.html', {'archivo_resultados': 'apps/predictions/data_set/predicciones_fraude_resultados_gb.csv', 'df_predicciones': df_resultados_preview})



''' Funcion para el modelo SVM '''
def predictions_svm(request):
    path_data = os.path.join(settings.BASE_DIR, 'apps/predictions/data_set/card_transdata.csv')
    path_model = os.path.join(settings.BASE_DIR, 'apps/predictions/models/svm_model.pkl')

    # Leer el archivo CSV
    df = pd.read_csv(path_data)


    # Cargar el modelo
    with open(path_model, 'rb') as model_file:
        svm_model = pickle.load(model_file)

    # Número total de registros en el DataFrame
    total_registros = len(df) - 995000

    # Tamaño del lote (batch)
    tamano_lote = 1000

    # Lista para almacenar los resultados de cada lote
    resultados = []

    # Iterar sobre lotes de 1000 registros
    for inicio in range(0, total_registros, tamano_lote):
        fin = min(inicio + tamano_lote, total_registros)
        
        # Tomar el lote actual
        lote_df = df.iloc[inicio:fin]

        # Asegurarse de tener las columnas necesarias para las predicciones
        # Puedes ajustar esto según las características utilizadas por tu modelo
        if 'fraud' in lote_df.columns:
            X_lote = lote_df.drop('fraud', axis=1)
        else:
            X_lote = lote_df.copy()

        # Hacer predicciones
        predictions = svm_model.predict(X_lote)

        # Agregar las predicciones al DataFrame del lote
        lote_df['predicted_fraud'] = predictions

        # Agregar el lote a la lista de resultados
        resultados.append(lote_df)

    # Concatenar todos los lotes en un solo DataFrame
    df_resultados = pd.concat(resultados, ignore_index=True)

    # Obtener solo las primeras 100 filas del DataFrame de resultados
    df_resultados_preview = df_resultados.head(100)

    # Guardar el DataFrame de resultados en un nuevo archivo CSV (opcional)
    df_resultados_preview.to_csv('apps/predictions/data_set/predicciones_fraude_resultados_preview.csv', index=False)

    # Guardar el DataFrame de resultados en un nuevo archivo CSV
    df_resultados.to_csv('apps/predictions/data_set/predicciones_fraude_resultados_svm.csv', index=False)

    # Renderizar un template con información sobre el procesamiento
    return render(request, 'predictions_svm.html', {'archivo_resultados': 'apps/predictions/data_set/predicciones_fraude_resultados_svm.csv', 'df_predicciones': df_resultados_preview})


''' Funcion para la descarga de archivos gboost '''
def predictions_gboost_download(request):
    filename = 'apps/predictions/data_set/predicciones_fraude_resultados_gb.csv'
    response = FileResponse(open(filename, 'rb'))
    content_type, encoding = mimetypes.guess_type(filename)
    response['Content-Type'] = content_type
    response['Content-Disposition'] = f'attachment; filename="{filename}"'
    return response


''' Funcion para la descarga de archivos nb '''
def predictions_nb_download(request):
    filename = 'apps/predictions/data_set/predicciones_fraude_resultados_nb.csv'
    response = FileResponse(open(filename, 'rb'))
    content_type, encoding = mimetypes.guess_type(filename)
    response['Content-Type'] = content_type
    response['Content-Disposition'] = f'attachment; filename="{filename}"'
    return response

''' Funcion para la descarga de archivos svm '''
def predictions_svm_download(request):
    filename = 'apps/predictions/data_set/predicciones_fraude_resultados_svm.csv'
    response = FileResponse(open(filename, 'rb'))
    content_type, encoding = mimetypes.guess_type(filename)
    response['Content-Type'] = content_type
    response['Content-Disposition'] = f'attachment; filename="{filename}"'
    return response