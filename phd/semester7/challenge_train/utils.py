# Función para cargar imágenes
def load_images_from_directory(directory, num_images=5):
    images = []
    filenames = os.listdir(directory)

    for i, file in enumerate(filenames):
        if i >= num_images:  # Limitar el número de imágenes a cargar
            break
        if file.endswith(".png"):
            img_path = os.path.join(directory, file)
            img = Image.open(img_path)
            images.append(np.array(img))

    return images


def display_images(figure_name, noisy_images, patch_images, num_images=5000):
    # Asegúrate de que no se superen los límites de las listas
    num_images = min(num_images, len(noisy_images), len(patch_images))

    for i in range(num_images):
        plt.figure(figsize=(10, 5))

        # Mostrar la imagen del parche correspondiente
        plt.subplot(1, 2, 1)
        plt.imshow(patch_images[i])
        plt.title("Patch Image")
        plt.axis("off")

        # Mostrar la imagen ruidosa
        plt.subplot(1, 2, 2)
        plt.imshow(noisy_images[i])
        plt.title("Noisy Image")
        plt.axis("off")

        plt.savefig(figure_name)
        plt.close()
