import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


class Ecualizador:

    def ecualizar(self, imagen: np.ndarray, kernel: tuple[int, int]) -> np.ndarray:

        img = imagen.copy()
        nueva_img = np.zeros_like(img, dtype=np.float32)

        #suavizado 
        img = cv2.GaussianBlur(img, (5, 5), 0)

        # calcular padding para que analice bien los bordes de la imagen
        mitad_w = kernel[0] // 2
        mitad_h = kernel[1] // 2

        # Extender la imagen con bordes replicados del padding
        img_pad = cv2.copyMakeBorder(
            img,
            mitad_h, mitad_h, mitad_w, mitad_w,
            cv2.BORDER_REPLICATE
        )

        print(f"Procesando con ventana {kernel[0]}x{kernel[1]}...", end=" ")

        for i in range(imagen.shape[0]):
            for j in range(imagen.shape[1]):
                # region de interes
                y_start = i
                y_end = i + 2 * mitad_h + 1
                x_start = j
                x_end = j + 2 * mitad_w + 1

                seccion = img_pad[y_start:y_end, x_start:x_end]

                # Calclar histograma
                histograma = cv2.calcHist([seccion], [0], None, [256], [0, 256])

                # Normalizar el histograma
                histograma_norm = histograma / (seccion.shape[0] * seccion.shape[1])

                # Calcular el CDF
                cdf = histograma_norm.cumsum()

                # Normalizar CDF a rango [0, 255]
                cdf_normalizado = cdf * 255

                # Mapear el pixel central usando la transformacion de ecualizacion
                pixel_original = img_pad[i + mitad_h, j + mitad_w]
                nueva_img[i, j] = cdf_normalizado[pixel_original]

        print("OK")
        return nueva_img.astype(np.uint8)

    def show_equalized(self, imagen: np.ndarray, kernel: tuple[int, int]):

        nueva_img = self.ecualizar(imagen, kernel)

        hist_original = cv2.calcHist([imagen], [0], None, [256], [0, 256])
        hist_ecualizada = cv2.calcHist([nueva_img], [0], None, [256], [0, 256])

        fig = plt.figure(figsize=(15, 5))

        # Subplot para la imagen ecualizada 
        ax0 = plt.subplot2grid((2, 3), (0, 0), colspan=2, rowspan=2)
        ax0.imshow(nueva_img, cmap="gray")
        ax0.set_title(f"Imagen Ecualizada - Ventana {kernel[0]}x{kernel[1]}", fontsize=12)
        ax0.axis('off')

        # Subplot para histograma original
        ax1 = plt.subplot2grid((2, 3), (0, 2))
        ax1.plot(hist_original, color='blue', linewidth=1.5)
        ax1.set_title("Histograma Original", fontsize=10)
        ax1.set_xlim([0, 255])
        ax1.grid(alpha=0.3)

        # Subplot para histograma ecualizado
        ax2 = plt.subplot2grid((2, 3), (1, 2))
        ax2.plot(hist_ecualizada, color='green', linewidth=1.5)
        ax2.set_title("Histograma Ecualizado", fontsize=10)
        ax2.set_xlim([0, 255])
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        plt.show()


def main():

    print("="*70)
    print("EJERCICIO 1: Ecualizacion Local de Histograma")
    print("="*70)
    print()

    # Cargar imagen
    img_path = Path(__file__).parent / "Imagen_con_detalles_escondidos.tif"

    if not img_path.exists():
        print(f"Error: No se encontro la imagen en {img_path}")
        print("Verifica que la imagen este en la misma carpeta que este script.")
        return

    print(f"Cargando imagen: {img_path.name}")
    imagen = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

    if imagen is None:
        print("Error: No se pudo cargar la imagen")
        return

    print(f"Dimensiones: {imagen.shape[0]}x{imagen.shape[1]}")
    print()

    # Crear ecualizador
    ecualizador = Ecualizador()

    # Definir diferentes tamaños de ventana para analizar
    kernels = [
        (5, 5),
        (11, 11),
        (11, 21),
        (21, 21),
        (51, 51),
        (111, 111),
    ]

    print("-" * 70)

    for kernel in kernels:
        ecualizador.show_equalized(imagen, kernel)

    print("-"*70)
    print()
    print("DETALLES ESCONDIDOS ENCONTRADOS:")
    print("  - Ventanas pequeñas (5x5, 11x11): Revelan textura fina y detalles locales")
    print("  - Ventanas medianas (21x21, 51x51): Balance entre detalle y uniformidad")
    print("  - Ventanas grandes (111x111): Efecto similar a ecualizacion global")
    print()
    print("CONCLUSIONES:")
    print("  * Ventanas pequeñas: Mas sensibles al ruido, mayor contraste local")
    print("  * Ventanas grandes: Menos detalle local, resultado mas suave")
    print("  * Ventana optima: Depende del tamaño de los detalles a revelar")
    print()


if __name__ == "__main__":
    main()
