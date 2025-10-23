import numpy as np
import cv2
import csv
import os
from pathlib import Path

def escanear_formulario(img):
    formulario = {}

    imgumbral = img <= 150
    imgfilas = np.diff(imgumbral, axis=0)
    imgfilassum = np.sum(imgfilas, axis=1)
    imgfilas_bool = imgfilassum > 900
    imgfilas_idxs = np.argwhere(imgfilas_bool).ravel()

    imgfilas_diff_idxs = np.diff(imgfilas_idxs)
    indices_a_eliminar = np.where(imgfilas_diff_idxs <= 10)[0]
    imgfilas_idxs_validos = np.delete(imgfilas_idxs, indices_a_eliminar)

    filas_y = np.vstack((imgfilas_idxs_validos[:-1], imgfilas_idxs_validos[1:])).T

    for i, item in enumerate(filas_y):
        fila = img[item[0]:item[1]]
        fila_umbral = fila <= 150

        imgcols = np.diff(fila_umbral, axis=1)
        imgcolssum = np.sum(imgcols, axis=0)
        imgcols_bool = imgcolssum >= 30
        imgcols_idxs = np.argwhere(imgcols_bool).ravel()

        imgcols_diff_idxs = np.diff(imgcols_idxs)
        indices_a_eliminar = np.where(imgcols_diff_idxs <= 2)[0]
        imgcols_idxs_validos = np.delete(imgcols_idxs, indices_a_eliminar)

        if len(imgcols_idxs_validos) < 2:
            continue

        columnas_x = np.vstack((imgcols_idxs_validos[:-1], imgcols_idxs_validos[1:])).T

        for j, columna in enumerate(columnas_x):
            ARRIBA = 2
            ABAJO = fila_umbral.shape[0] - 1
            IZQUIERDA = columna[0] + 1
            DERECHA = columna[1]
            img_col = fila_umbral[ARRIBA:ABAJO, IZQUIERDA:DERECHA] * 255

            if i == 0:
                continue
            elif i == 1 and j == 1:
                formulario["nombre"] = img_col
            elif i == 2 and j == 1:
                formulario["edad"] = img_col
            elif i == 3 and j == 1:
                formulario["mail"] = img_col
            elif i == 4 and j == 1:
                formulario["legajo"] = img_col
            elif i == 6:
                if j == 1:
                    formulario["preg1_si"] = img_col
                elif j == 2:
                    formulario["preg1_no"] = img_col
            elif i == 7:
                if j == 1:
                    formulario["preg2_si"] = img_col
                elif j == 2:
                    formulario["preg2_no"] = img_col
            elif i == 8:
                if j == 1:
                    formulario["preg3_si"] = img_col
                elif j == 2:
                    formulario["preg3_no"] = img_col
            elif i == 9 and j == 1:
                formulario["comentario"] = img_col

    return formulario

def contar_valores_consecutivos(arr):

    if len(arr) == 0:
        return []

    resultado = []
    valor_actual = arr[0]
    conteo_actual = 1

    for i in range(1, len(arr)):
        if arr[i] == valor_actual:
            conteo_actual += 1
        else:
            resultado.append((valor_actual, conteo_actual))
            valor_actual = arr[i]
            conteo_actual = 1

    resultado.append((valor_actual, conteo_actual))
    return resultado


def reemplazar_false_consecutivo(arr, umbral):

    if len(arr) == 0:
        return arr

    arrm = contar_valores_consecutivos(arr)

    if len(arrm) == 0:
        return arr

    conteos_nuevos = []

    primero = arrm[0]
    ultimo = arrm[-1]

    for i in range(1, len(arrm) - 1):
        valor, conteo = arrm[i]
        if not valor and conteo < umbral:
            conteos_nuevos.append((True, conteo))
        else:
            conteos_nuevos.append(arrm[i])

    resultado = [primero] + conteos_nuevos + [ultimo]
    resultado_array = np.array([valor for valor, conteo in resultado for _ in range(conteo)])
    return resultado_array


def conteo_parrafos(arr):
    arrm = contar_valores_consecutivos(arr)
    conteo_parrafos = [conteo for valor, conteo in arrm if valor]
    return len(conteo_parrafos)

def contar_elementos(img, axis, umbral):
    img_zeros = img == 0
    img_sum = img_zeros.any(axis=axis)

    modificado = reemplazar_false_consecutivo(img_sum, umbral)
    num_elementos = conteo_parrafos(modificado)

    return num_elementos

def validar_texto(img, car_min=0, car_max=float('inf'), palabras_min=0, palabras_max=float('inf'), axis=0, car_umbral=1, palabra_umbral=10):
    if img is None or img.size == 0:
        return False

    num_palabras = contar_elementos(img, axis, palabra_umbral)
    num_car = contar_elementos(img, axis, car_umbral) + num_palabras - 1

    if car_min <= num_car <= car_max and palabras_min <= num_palabras <= palabras_max:
        return True
    else:
        return False

def validar_formulario(campos):

    resultados = {}

    criteria = {
        'nombre': {'palabras_min': 2, 'palabras_max': float('inf'), 'car_min': 1, 'car_max': 25},
        'edad': {'palabras_min': 1, 'palabras_max': 1, 'car_min': 2, 'car_max': 3},
        'mail': {'palabras_min': 1, 'palabras_max': 1, 'car_min': 1, 'car_max': 25},
        'legajo': {'palabras_min': 1, 'palabras_max': 1, 'car_min': 8, 'car_max': 8},
        'preg': {'palabras_min': 1, 'palabras_max': 1, 'car_min': 1, 'car_max': 1},
        'comentario': {'palabras_min': 1, 'palabras_max': float('inf'), 'car_min': 1, 'car_max': 25},
    }

    nombre = campos.get("nombre")
    resultados["Nombre y apellido"] = 'OK' if validar_texto(nombre == 0, **criteria['nombre']) else 'MAL'

    edad = campos.get("edad")
    resultados["Edad"] = 'OK' if validar_texto(edad == 0, **criteria['edad']) else 'MAL'

    mail = campos.get("mail")
    resultados["Mail"] = 'OK' if validar_texto(mail == 0, **criteria['mail']) else 'MAL'

    legajo = campos.get("legajo")
    resultados["Legajo"] = 'OK' if validar_texto(legajo == 0, **criteria['legajo']) else 'MAL'

    si1 = validar_texto(campos.get("preg1_si") == 0, **criteria['preg'])
    no1 = validar_texto(campos.get("preg1_no") == 0, **criteria['preg'])
    resultados["Pregunta 1"] = 'OK' if (si1 and not no1) or (no1 and not si1) else 'MAL'

    si2 = validar_texto(campos.get("preg2_si") == 0, **criteria['preg'])
    no2 = validar_texto(campos.get("preg2_no") == 0, **criteria['preg'])
    resultados["Pregunta 2"] = 'OK' if (si2 and not no2) or (no2 and not si2) else 'MAL'

    si3 = validar_texto(campos.get("preg3_si") == 0, **criteria['preg'])
    no3 = validar_texto(campos.get("preg3_no") == 0, **criteria['preg'])
    resultados["Pregunta 3"] = 'OK' if (si3 and not no3) or (no3 and not si3) else 'MAL'

    comentario = campos.get("comentario")
    resultados["Comentarios"] = 'OK' if validar_texto(comentario == 0, **criteria['comentario']) else 'MAL'

    return resultados

def extraer_tipo_formulario(img):

    encabezado = img[5:45, :]
    umbral = 150
    img_binaria = (encabezado < umbral).astype(np.uint8)
    img_binaria_uint8 = img_binaria * 255
    num_etiquetas, etiquetas, stats, _ = cv2.connectedComponentsWithStats(img_binaria, 8, cv2.CV_32S)

    area_minima = 45
    area_maxima = 600 

    componentes_validas = []
    for i in range(1, num_etiquetas):
        area = stats[i, cv2.CC_STAT_AREA]
        ancho = stats[i, cv2.CC_STAT_WIDTH]
        alto = stats[i, cv2.CC_STAT_HEIGHT]
        ratio = ancho / alto if alto > 0 else 0

        # Filtro más permisivo para la altura
        if area_minima < area < area_maxima and ratio < 3.5 and alto > 7:
            componentes_validas.append(i)

    if len(componentes_validas) == 0:
        print("  [ADVERTENCIA] No se detectaron componentes válidas")
        return 'desconocido'

    # elegir la letra del tipo
    componentes_ordenadas = sorted(componentes_validas,
                                   key=lambda i: stats[i, cv2.CC_STAT_LEFT])
    idx_letra = componentes_ordenadas[-1]
    x = stats[idx_letra, cv2.CC_STAT_LEFT]
    y = stats[idx_letra, cv2.CC_STAT_TOP]
    ancho = stats[idx_letra, cv2.CC_STAT_WIDTH]
    alto = stats[idx_letra, cv2.CC_STAT_HEIGHT]
    area = stats[idx_letra, cv2.CC_STAT_AREA]
    ratio = ancho / alto if alto > 0 else 0
    letra_mask = (etiquetas == idx_letra).astype(np.uint8)
    letra_region = letra_mask[y:y+alto, x:x+ancho]
    perfil_horizontal = np.sum(letra_region, axis=1).astype(float)
    perfil_vertical = np.sum(letra_region, axis=0).astype(float)

    #Normalizamos perfil horizontal (de arriba hacia abajo)
    if np.max(perfil_horizontal) > 0:
        perfil_h_norm = perfil_horizontal / np.max(perfil_horizontal)
    else:
        return 'desconocido'

    #Normalizamos perfil vertical (de izquierda a derecha)
    if np.max(perfil_vertical) > 0:
        perfil_v_norm = perfil_vertical / np.max(perfil_vertical)
    else:
        return 'desconocido'

    # Dividir perfil horizontal en tercios
    tercio = len(perfil_h_norm) // 3
    tercio_superior = np.mean(perfil_h_norm[:tercio]) if tercio > 0 else 0
    tercio_medio = np.mean(perfil_h_norm[tercio:2*tercio]) if tercio > 0 else 0
    tercio_inferior = np.mean(perfil_h_norm[2*tercio:]) if tercio > 0 else 0

    varianza_h = np.var(perfil_h_norm)

    ratio_inferior_superior = tercio_inferior / tercio_superior if tercio_superior > 0.1 else 1

    if ratio_inferior_superior >= 1.15:
        tipo = 'A'
    elif ratio_inferior_superior >= 0.4:
        tipo = 'B'
    else:
        tipo = 'C'

    return tipo



#Procesamos el formulario

def procesar_formulario(ruta_imagen):

    img = cv2.imread(str(ruta_imagen), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, None, None
    tipo_formulario = extraer_tipo_formulario(img) 
    campos = escanear_formulario(img)
    resultados = validar_formulario(campos)

    return resultados, campos, tipo_formulario



def procesar_lote(directorio='.'):

    directorio = Path(directorio)
    archivos = sorted(directorio.glob("formulario_*.png"))

    todos_resultados = []
    lote_para_imagen = []

    resultados_por_tipo = {'A': [], 'B': [], 'C': [], 'desconocido': [], None: []}

    for archivo in archivos:
        print(f"Procesando: {archivo.name}")
        resultados, campos, tipo_formulario = procesar_formulario(archivo)

        if resultados is None:
            continue

        for campo, resultado in resultados.items():
            print(f"  > {campo}: {resultado}")
        print(f"  > Tipo de formulario: {tipo_formulario}\n")

        id_form = archivo.stem.replace("formulario_", "")

        fila = [id_form]
        for campo in ["Nombre y apellido", "Edad", "Mail", "Legajo",
                      "Pregunta 1", "Pregunta 2", "Pregunta 3", "Comentarios"]:
            fila.append(resultados.get(campo, 'MAL'))

        todos_resultados.append(fila)

        es_correcto = all(v == 'OK' for v in resultados.values())
        lote_para_imagen.append((id_form, tipo_formulario, es_correcto, campos.get('nombre')))

        resultados_por_tipo[tipo_formulario].append({
            'id': id_form,
            'correcto': es_correcto
        })

    print("Descripción de cada formulario")
    for tipo in ['A', 'B', 'C']:
        formularios = resultados_por_tipo[tipo]
        if formularios:
            correctos = sum(1 for f in formularios if f['correcto'])
            print(f"\nFormulario {tipo}:")
            print(f"  Total: {len(formularios)}")
            print(f"  Correctos: {correctos}")
            print(f"  Incorrectos: {len(formularios) - correctos}")
            for form in formularios:
                estado = "OK" if form['correcto'] else "MAL"
                print(f"    - ID {form['id']}: {estado}")

    return todos_resultados, lote_para_imagen


def generar_imagen_salida(lote_formularios, directorio_salida='salida'):

    if not os.path.exists(directorio_salida):
        os.makedirs(directorio_salida)

    # Dimensiones
    ancho_recuadro = 200
    alto_recuadro = 100
    espaciado = 20

    cols = 3
    filas = (len(lote_formularios) + cols - 1) // cols

    ancho_total = cols * ancho_recuadro + (cols + 1) * espaciado
    alto_total = filas * alto_recuadro + (filas + 1) * espaciado + 50

    #Crear imagen blanca
    img_salida = np.ones((alto_total, ancho_total, 3), dtype=np.uint8) * 255

    #Título
    cv2.putText(img_salida, 'Validacion de Formularios', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    y_inicio = 60

    for idx, (id_form, tipo_form, es_correcto, nombre_img) in enumerate(lote_formularios):
        col = idx % cols
        fila = idx // cols

        x = espaciado + col * (ancho_recuadro + espaciado)
        y = y_inicio + fila * (alto_recuadro + espaciado)

        #Color según resultado
        if es_correcto:
            color = (0, 255, 0)  #Verde (bien)
            texto_estado = "OK"
        else:
            color = (0, 0, 255)  #Rojo (mal)
            texto_estado = "MAL"

        #Dibuja el recuadro
        cv2.rectangle(img_salida, (x, y), (x + ancho_recuadro, y + alto_recuadro), color, 3)

        #Texto
        cv2.putText(img_salida, f"ID: {id_form}", (x + 10, y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(img_salida, f"Tipo: {tipo_form}", (x + 10, y + 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(img_salida, texto_estado, (x + 10, y + 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    ruta_salida = os.path.join(directorio_salida, 'validacion_formularios.png')
    cv2.imwrite(ruta_salida, img_salida)
    print(f"\nImagen de salida guardada en: {ruta_salida}")


def guardar_csv(resultados, archivo_salida='validaciones.csv'):

    headers = ['ID', 'Nombre y apellido', 'Edad', 'Mail', 'Legajo',
               'Pregunta 1', 'Pregunta 2', 'Pregunta 3', 'Comentarios']

    with open(archivo_salida, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(resultados)

    print(f"Resultados guardados en: {archivo_salida}")


if __name__ == "__main__":
    resultados, lote_para_imagen = procesar_lote()
    guardar_csv(resultados)
    generar_imagen_salida(lote_para_imagen)
