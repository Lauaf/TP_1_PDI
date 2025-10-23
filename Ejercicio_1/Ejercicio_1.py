import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def ecualizar(imagen: np.ndarray, kernel: tuple[int, int]) -> np.ndarray:
    img = imagen.copy()
    nueva_img = np.zeros_like(img, dtype=np.float32)

    # Suavizado para bajarle el ruido
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # Pading para bordes
    mitad_w = kernel[0] // 2
    mitad_h = kernel[1] // 2

    img_pad = cv2.copyMakeBorder(
        img,
        mitad_h, mitad_h, mitad_w, mitad_w,
        cv2.BORDER_REPLICATE,
    )

    print(f"Procesando con ventana {kernel[0]}x{kernel[1]}...", end=" ")

    for i in range(imagen.shape[0]):
        for j in range(imagen.shape[1]):
            # Región que va a usar el histograma
            y_start = i
            y_end = i + 2 * mitad_h + 1
            x_start = j
            x_end = j + 2 * mitad_w + 1

            seccion = img_pad[y_start:y_end, x_start:x_end]

            # Histograma local (256 bins)
            histograma = cv2.calcHist([seccion], [0], None, [256], [0, 256])

            # Normalización + CDF
            histograma_norm = histograma / (seccion.shape[0] * seccion.shape[1])
            cdf = histograma_norm.cumsum()
            cdf_normalizado = cdf * 255

            # Mapear el píxel central mediante CDF
            pixel_original = img_pad[i + mitad_h, j + mitad_w]
            nueva_img[i, j] = cdf_normalizado[pixel_original]

    print("OK")
    return nueva_img.astype(np.uint8)


def mostrar_ecualizada(imagen: np.ndarray, kernel: tuple[int, int]) -> None:
    nueva_img = ecualizar(imagen, kernel)

    hist_original = cv2.calcHist([imagen], [0], None, [256], [0, 256])
    hist_ecualizada = cv2.calcHist([nueva_img], [0], None, [256], [0, 256])

    plt.figure(figsize=(15, 5))

    # Imagen ecualizada
    ax0 = plt.subplot2grid((2, 3), (0, 0), colspan=2, rowspan=2)
    ax0.imshow(nueva_img, cmap="gray")
    ax0.set_title(f"Imagen Ecualizada - Ventana {kernel[0]}x{kernel[1]}", fontsize=12)
    ax0.axis("off")

    # Histograma original
    ax1 = plt.subplot2grid((2, 3), (0, 2))
    ax1.plot(hist_original, color="blue", linewidth=1.5)
    ax1.set_title("Histograma Original", fontsize=10)
    ax1.set_xlim([0, 255])
    ax1.grid(alpha=0.3)

    # Histograma ecualizado
    ax2 = plt.subplot2grid((2, 3), (1, 2))
    ax2.plot(hist_ecualizada, color="green", linewidth=1.5)
    ax2.set_title("Histograma Ecualizado", fontsize=10)
    ax2.set_xlim([0, 255])
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def main():
    print("=" * 45)
    print("EJERCICIO 1: Ecualización Local de Histograma")
    print("=" * 45)
    print()

    img_path = Path(__file__).parent / "Imagen_con_detalles_escondidos.tif"

    if not img_path.exists():
        print(f"Error: No se encontró la imagen en {img_path}")
        return

    print(f"Cargando imagen: {img_path.name}")
    imagen = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

    # Diferentes tamaños de ventans de analisis
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
        mostrar_ecualizada(imagen, kernel)

    #
    print("-" * 70)
    print()
    print(" Informe de los detalles escondidos :")
    print("  - Ventana arriba-izquierda :Se revela un cuadrado")
    print("  - Ventana arriba-derecha :Se revela una linea diagonal")
    print("  - Ventana centro :Se revela una letra 'a'")
    print("  - Ventana abajo-izquierda :Se revelan 4 lineas horizontales apiladas verticalmente")
    print("  - Ventana abajo-derecha :Se revela un circulo")
    print()
    print("CONCLUSIONES:")
    print("  * Ventanas pequeñas (5x5, 11x11): Más sensibles al ruido, mayor contraste local pero revelan textura fina y detalles locales")
    print("  * Ventanas grandes (51x51, 111x111): Menos detalle local y resultado más suave, efecto similar a ecualización global")
    print("  * Ventana óptima (11x21, 21x21): Balance entre detalle y uniformidad pero depende del tamaño de los detalles a revelar")
    print()


if __name__ == "__main__":
    main()

