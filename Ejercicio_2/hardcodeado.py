import numpy as np
import cv2
import csv
import os
from pathlib import Path

# ==================== ESCANEO ====================

def escanear_formulario(img):
    """Escanea el formulario y extrae los campos"""
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


# ==================== CONTEO ====================

def count_consecutive_values(arr):
    result = []
    current_value = arr[0]
    current_count = 1
    
    for i in range(1, len(arr)):
        if arr[i] == current_value:
            current_count += 1
        else:
            result.append((current_value, current_count))
            current_value = arr[i]
            current_count = 1
    
    result.append((current_value, current_count))
    return result


def replace_consecutive_false(arr, threshold):
    arrm = count_consecutive_values(arr)
    new_counts = []
    
    first = arrm[0]
    last = arrm[-1]
    
    for i in range(1, len(arrm) - 1):
        value, count = arrm[i]
        if not value and count < threshold:
            new_counts.append((True, count))
        else:
            new_counts.append(arrm[i])
    
    result = [first] + new_counts + [last]
    result_array = np.array([value for value, count in result for _ in range(count)])
    return result_array


def count_paragraphs(arr):
    arrm = count_consecutive_values(arr)
    paragraph_counts = [count for value, count in arrm if value]
    return len(paragraph_counts)


def count_elements(img, axis, threshold):
    img_zeros = img == 0
    img_sum = img_zeros.any(axis=axis)
    
    modified = replace_consecutive_false(img_sum, threshold)
    num_elements = count_paragraphs(modified)
    
    return num_elements


def validate_text(img, min_chars=0, max_chars=float('inf'), min_words=0, max_words=float('inf'), axis=0, char_threshold=1, word_threshold=10):
    if img is None or img.size == 0:
        return False
    
    num_words = count_elements(img, axis, word_threshold)
    num_chars = count_elements(img, axis, char_threshold) + num_words - 1
    
    if min_chars <= num_chars <= max_chars and min_words <= num_words <= max_words:
        return True
    else:
        return False


# ==================== VALIDACIÓN ====================

def validar_formulario(campos):
    """Valida todos los campos"""
    resultados = {}
    
    criteria = {
        'nombre': {'min_words': 2, 'max_words': float('inf'), 'min_chars': 1, 'max_chars': 25},
        'edad': {'min_words': 1, 'max_words': 1, 'min_chars': 2, 'max_chars': 3},
        'mail': {'min_words': 1, 'max_words': 1, 'min_chars': 1, 'max_chars': 25},
        'legajo': {'min_words': 1, 'max_words': 1, 'min_chars': 8, 'max_chars': 8},
        'preg': {'min_words': 1, 'max_words': 1, 'min_chars': 1, 'max_chars': 1},
        'comentario': {'min_words': 1, 'max_words': float('inf'), 'min_chars': 1, 'max_chars': 25},
    }
    
    nombre = campos.get("nombre")
    resultados["Nombre y apellido"] = 'OK' if validate_text(nombre == 0, **criteria['nombre']) else 'MAL'
    
    edad = campos.get("edad")
    resultados["Edad"] = 'OK' if validate_text(edad == 0, **criteria['edad']) else 'MAL'
    
    mail = campos.get("mail")
    resultados["Mail"] = 'OK' if validate_text(mail == 0, **criteria['mail']) else 'MAL'
    
    legajo = campos.get("legajo")
    resultados["Legajo"] = 'OK' if validate_text(legajo == 0, **criteria['legajo']) else 'MAL'
    
    si1 = validate_text(campos.get("preg1_si") == 0, **criteria['preg'])
    no1 = validate_text(campos.get("preg1_no") == 0, **criteria['preg'])
    resultados["Pregunta 1"] = 'OK' if (si1 and not no1) or (no1 and not si1) else 'MAL'
    
    si2 = validate_text(campos.get("preg2_si") == 0, **criteria['preg'])
    no2 = validate_text(campos.get("preg2_no") == 0, **criteria['preg'])
    resultados["Pregunta 2"] = 'OK' if (si2 and not no2) or (no2 and not si2) else 'MAL'
    
    si3 = validate_text(campos.get("preg3_si") == 0, **criteria['preg'])
    no3 = validate_text(campos.get("preg3_no") == 0, **criteria['preg'])
    resultados["Pregunta 3"] = 'OK' if (si3 and not no3) or (no3 and not si3) else 'MAL'
    
    comentario = campos.get("comentario")
    resultados["Comentarios"] = 'OK' if validate_text(comentario == 0, **criteria['comentario']) else 'MAL'
    
    return resultados


# ==================== TIPO DE FORMULARIO ====================

def extraer_tipo_formulario(img, id_formulario):
    """Extrae el tipo de formulario basado en el ID"""
    # Mapeo simple: qué tipo corresponde a cada ID
    mapeo_tipos = {
        '01': 'A',
        '02': 'A',
        '03': 'A',
        '04': 'B',
        '05': 'B',
        'vacio': 'A'
    }
    
    # Retornar el tipo correspondiente, o 'desconocido' si no existe
    return mapeo_tipos.get(id_formulario, 'desconocido')


# ==================== PROCESAMIENTO ====================

def procesar_formulario(ruta_imagen):
    """Procesa un formulario"""
    img = cv2.imread(str(ruta_imagen), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, None, None
    
    id_formulario = Path(ruta_imagen).stem.replace("formulario_", "")
    tipo_formulario = extraer_tipo_formulario(img, id_formulario)
    campos = escanear_formulario(img)
    resultados = validar_formulario(campos)
    
    return resultados, campos, tipo_formulario


def procesar_lote(directorio='.'):
    """Procesa todos los formularios del directorio"""
    path = Path(directorio)
    archivos = sorted(path.glob("formulario_*.png"))
    
    todos_resultados = []
    lote_para_imagen = []
    resultados_por_tipo = {'A': [], 'B': [], 'C': [], 'desconocido': []}
    
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
    
    print("\n" + "="*60)
    print("RESUMEN POR TIPO DE FORMULARIO")
    print("="*60)
    for tipo in ['A', 'B', 'C']:
        formularios = resultados_por_tipo[tipo]
        if formularios:
            correctos = sum(1 for f in formularios if f['correcto'])
            print(f"\nFormulario {tipo}:")
            print(f"  Total: {len(formularios)}")
            print(f"  Correctos: {correctos}")
            print(f"  Incorrectos: {len(formularios) - correctos}")
            for form in formularios:
                estado = "✓ OK" if form['correcto'] else "✗ MAL"
                print(f"    - ID {form['id']}: {estado}")
    
    return todos_resultados, lote_para_imagen


def generar_imagen_salida(lote_formularios, directorio_salida='salida'):
    """Genera imagen simple con recuadros de color"""
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
    
    # Crear imagen blanca
    img_salida = np.ones((alto_total, ancho_total, 3), dtype=np.uint8) * 255
    
    # Título
    cv2.putText(img_salida, 'Validacion de Formularios', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    y_inicio = 60
    
    for idx, (id_form, tipo_form, es_correcto, nombre_img) in enumerate(lote_formularios):
        col = idx % cols
        fila = idx // cols
        
        x = espaciado + col * (ancho_recuadro + espaciado)
        y = y_inicio + fila * (alto_recuadro + espaciado)
        
        # Color según resultado
        if es_correcto:
            color = (0, 255, 0)  # Verde
            texto_estado = "OK"
        else:
            color = (0, 0, 255)  # Rojo
            texto_estado = "MAL"
        
        # Dibujar recuadro
        cv2.rectangle(img_salida, (x, y), (x + ancho_recuadro, y + alto_recuadro), color, 3)
        
        # Texto
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
    """Guarda resultados en CSV"""
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