import numpy as np
import cv2
import csv
import os
from pathlib import Path

def escanear_formulario(img):
    formulario = {}

    img_umbralizada = img <= 150
    img_filas = np.diff(img_umbralizada, axis=0)
    img_filassum = np.sum(img_filas, axis=1)
    img_filas_bool = img_filassum > 900
    img_filas_idxs = np.argwhere(img_filas_bool).ravel()

    img_filas_diff_idxs = np.diff(img_filas_idxs)
    indices_a_eliminar = np.where(img_filas_diff_idxs <= 10)[0]
    img_filas_idxs_validos = np.delete(img_filas_idxs, indices_a_eliminar)

    rangos_filas = np.vstack((img_filas_idxs_validos[:-1], img_filas_idxs_validos[1:])).T

    for i, item in enumerate(rangos_filas):
        fila = img[item[0]:item[1]]
        fila_umbral = fila <= 150

        diferencia_columnas = np.diff(fila_umbral, axis=1)
        suma_columnas = np.sum(diferencia_columnas, axis=0)
        columnas_detectadas = suma_columnas >= 30
        indices_columnas = np.argwhere(columnas_detectadas).ravel()

        diferencia_indices_col = np.diff(indices_columnas)
        indices_a_eliminar = np.where(diferencia_indices_col <= 2)[0]
        indices_validos_col = np.delete(indices_columnas, indices_a_eliminar)


        if len(indices_validos_col) < 2:
            continue

        rangos_columnas = np.vstack((indices_validos_col[:-1], indices_validos_col[1:])).T


        for j, columna in enumerate(rangos_columnas):
            ARRIBA = 2
            ABAJO = fila_umbral.shape[0] - 1
            IZQUIERDA = columna[0] + 1
            DERECHA = columna[1]
            campo_img = fila_umbral[ARRIBA:ABAJO, IZQUIERDA:DERECHA] * 255

            if i == 0:
                continue
            elif i == 1 and j == 1:
                formulario["nombre"] = campo_img
            elif i == 2 and j == 1:
                formulario["edad"] = campo_img
            elif i == 3 and j == 1:
                formulario["mail"] = campo_img
            elif i == 4 and j == 1:
                formulario["legajo"] = campo_img
            elif i == 6:
                if j == 1:
                    formulario["preg1_si"] = campo_img
                elif j == 2:
                    formulario["preg1_no"] = campo_img
            elif i == 7:
                if j == 1:
                    formulario["preg2_si"] = campo_img
                elif j == 2:
                    formulario["preg2_no"] = campo_img
            elif i == 8:
                if j == 1:
                    formulario["preg3_si"] = campo_img
                elif j == 2:
                    formulario["preg3_no"] = campo_img
            elif i == 9 and j == 1:
                formulario["comentario"] = campo_img

    return formulario

def contar_valores_consecutivos(arr):

    if len(arr) == 0:
        return []

    resultado = []
    valor_actual = arr[0]
    contador = 1

    for i in range(1, len(arr)):
        if arr[i] == valor_actual:
            contador += 1
        else:
            resultado.append((valor_actual, contador))
            valor_actual = arr[i]
            contador = 1

    resultado.append((valor_actual, contador))
    return resultado


def reemplazar_false_consecutivo(arr, umbral):

    if len(arr) == 0:
        return arr

    grupos = contar_valores_consecutivos(arr)

    if len(grupos) == 0:
        return arr

    nuevos_grupos = []

    primer_grupo = grupos[0]
    ultimo_grupo= grupos[-1]

    for i in range(1, len(grupos) - 1):
        valor, conteo = grupos[i]
        if not valor and conteo < umbral:
            nuevos_grupos.append((True, conteo))
        else:
            nuevos_grupos.append(grupos[i])

    resultado = [primer_grupo] + nuevos_grupos + [ultimo_grupo]
    array_limpio = np.array([valor for valor, conteo in resultado for _ in range(conteo)])
    return array_limpio


def cantidad_palabras(arr):
    grupos = contar_valores_consecutivos(arr)
    cantidad_palabras = [conteo for valor, conteo in grupos if valor]
    return len(cantidad_palabras)

def contar_elementos(img, eje, umbral):
    pixeles_negros = img == 0
    proyeccion = pixeles_negros.any(axis=eje)

    proyeccion_limpia = reemplazar_false_consecutivo(proyeccion, umbral)
    cantidad = cantidad_palabras(proyeccion_limpia)

    return cantidad

def validar_texto(img, min_caracteres=0, max_caracteres=float('inf'), min_palabras=0, max_palabras=float('inf'), eje=0, umbral_caracteres=1, umbral_palabras=10):
    if img is None or img.size == 0:
        return False

    cantidad_palabras = contar_elementos(img, eje, umbral_palabras)
    cantidad_caracteres = contar_elementos(img, eje, umbral_caracteres) + cantidad_palabras - 1

    if min_caracteres <= cantidad_caracteres <= max_caracteres and min_palabras <= cantidad_palabras <= max_palabras:
        return True
    else:
        return False

def validar_formulario(campos):

    resultados = {}

    criterios = {
        'nombre': {'min_palabras': 2, 'max_palabras': float('inf'), 'min_caracteres': 1, 'max_caracteres': 25},
        'edad': {'min_palabras': 1, 'max_palabras': 1, 'min_caracteres': 2, 'max_caracteres': 3},
        'mail': {'min_palabras': 1, 'max_palabras': 1, 'min_caracteres': 1, 'max_caracteres': 25},
        'legajo': {'min_palabras': 1, 'max_palabras': 1, 'min_caracteres': 8, 'max_caracteres': 8},
        'preg': {'min_palabras': 1, 'max_palabras': 1, 'min_caracteres': 1, 'max_caracteres': 1},
        'comentario': {'min_palabras': 1, 'max_palabras': float('inf'), 'min_caracteres': 1, 'max_caracteres': 25},
    }

    nombre = campos.get("nombre")
    resultados["Nombre y apellido"] = 'OK' if validar_texto(nombre == 0, **criterios['nombre']) else 'MAL'

    edad = campos.get("edad")
    resultados["Edad"] = 'OK' if validar_texto(edad == 0, **criterios['edad']) else 'MAL'

    mail = campos.get("mail")
    resultados["Mail"] = 'OK' if validar_texto(mail == 0, **criterios['mail']) else 'MAL'

    legajo = campos.get("legajo")
    resultados["Legajo"] = 'OK' if validar_texto(legajo == 0, **criterios['legajo']) else 'MAL'

    si1 = validar_texto(campos.get("preg1_si") == 0, **criterios['preg'])
    no1 = validar_texto(campos.get("preg1_no") == 0, **criterios['preg'])
    resultados["Pregunta 1"] = 'OK' if (si1 and not no1) or (no1 and not si1) else 'MAL'

    si2 = validar_texto(campos.get("preg2_si") == 0, **criterios['preg'])
    no2 = validar_texto(campos.get("preg2_no") == 0, **criterios['preg'])
    resultados["Pregunta 2"] = 'OK' if (si2 and not no2) or (no2 and not si2) else 'MAL'

    si3 = validar_texto(campos.get("preg3_si") == 0, **criterios['preg'])
    no3 = validar_texto(campos.get("preg3_no") == 0, **criterios['preg'])
    resultados["Pregunta 3"] = 'OK' if (si3 and not no3) or (no3 and not si3) else 'MAL'

    comentario = campos.get("comentario")
    resultados["Comentarios"] = 'OK' if validar_texto(comentario == 0, **criterios['comentario']) else 'MAL'

    return resultados

def extraer_tipo_formulario(img):

    encabezado = img[5:45, :]
    umbral = 150
    img_binaria = (encabezado < umbral).astype(np.uint8)
    img_binaria_uint8 = img_binaria * 255
    num_etiquetas, etiquetas, estadisticas, _ = cv2.connectedComponentsWithStats(img_binaria, 8, cv2.CV_32S)

    area_minima = 45
    area_maxima = 600 

    componentes_validas = []
    for i in range(1, num_etiquetas):
        area = estadisticas[i, cv2.CC_STAT_AREA]
        ancho = estadisticas[i, cv2.CC_STAT_WIDTH]
        alto = estadisticas[i, cv2.CC_STAT_HEIGHT]
        ratio = ancho / alto if alto > 0 else 0

        # Filtro más permisivo para la altura
        if area_minima < area < area_maxima and ratio < 3.5 and alto > 7:
            componentes_validas.append(i)

    if len(componentes_validas) == 0:
        print("  [ADVERTENCIA] No se detectaron componentes válidas")
        return 'desconocido'

    #Elegimos la letra del tipo
    componentes_ordenadas = sorted(componentes_validas,
                                   key=lambda i: estadisticas[i, cv2.CC_STAT_LEFT])
    indice_letra = componentes_ordenadas[-1]
    x = estadisticas[indice_letra, cv2.CC_STAT_LEFT]
    y = estadisticas[indice_letra, cv2.CC_STAT_TOP]
    ancho = estadisticas[indice_letra, cv2.CC_STAT_WIDTH]
    alto = estadisticas[indice_letra, cv2.CC_STAT_HEIGHT]
    area = estadisticas[indice_letra, cv2.CC_STAT_AREA]
    ratio = ancho / alto if alto > 0 else 0
    mascara_letra = (etiquetas == indice_letra).astype(np.uint8)
    region_letra = mascara_letra[y:y+alto, x:x+ancho]
    perfil_horizontal = np.sum(region_letra, axis=1).astype(float)
    perfil_vertical = np.sum(region_letra, axis=0).astype(float)

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

    relacion_inferior_superior = tercio_inferior / tercio_superior if tercio_superior > 0.1 else 1

    if relacion_inferior_superior >= 1.15:
        tipo = 'A'
    elif relacion_inferior_superior >= 0.4:
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

        formulario_correcto = all(v == 'OK' for v in resultados.values())
        lote_para_imagen.append((id_form, tipo_formulario, formulario_correcto, campos.get('nombre')))

        resultados_por_tipo[tipo_formulario].append({
            'id': id_form,
            'correcto': formulario_correcto
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

    columnas = 3
    filas = (len(lote_formularios) + columnas - 1) // columnas

    ancho_total = columnas * ancho_recuadro + (columnas + 1) * espaciado
    alto_total = filas * alto_recuadro + (filas + 1) * espaciado + 50

    #Crear imagen blanca
    img_salida = np.ones((alto_total, ancho_total, 3), dtype=np.uint8) * 255

    #Título
    cv2.putText(img_salida, 'Validacion de Formularios', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    y_inicio = 60

    for idx, (id_form, tipo_form, formulario_correcto, nombre_img) in enumerate(lote_formularios):
        col = idx % columnas
        fila = idx // columnas

        x = espaciado + col * (ancho_recuadro + espaciado)
        y = y_inicio + fila * (alto_recuadro + espaciado)

        #Color según resultado
        if formulario_correcto:
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

    encabezados = ['ID', 'Nombre y apellido', 'Edad', 'Mail', 'Legajo',
               'Pregunta 1', 'Pregunta 2', 'Pregunta 3', 'Comentarios']

    with open(archivo_salida, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(encabezados)
        writer.writerows(resultados)

    print(f"Resultados guardados en: {archivo_salida}")


if __name__ == "__main__":
    resultados, lote_para_imagen = procesar_lote()
    guardar_csv(resultados)
    generar_imagen_salida(lote_para_imagen)
