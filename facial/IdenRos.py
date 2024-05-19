import cv2
import os

def create_directory(path):
    if not os.path.exists(path):
        print(f'Carpeta creada: {path}')
        os.makedirs(path)

def draw_corner_rect(image, x, y, w, h, color, thickness, corner_length):
    # Dibujar esquinas en las cuatro direcciones
    cv2.line(image, (x, y), (x + corner_length, y), color, thickness)
    cv2.line(image, (x, y), (x, y + corner_length), color, thickness)
    
    cv2.line(image, (x + w, y), (x + w - corner_length, y), color, thickness)
    cv2.line(image, (x + w, y), (x + w, y + corner_length), color, thickness)
    
    cv2.line(image, (x, y + h), (x + corner_length, y + h), color, thickness)
    cv2.line(image, (x, y + h), (x, y + h - corner_length), color, thickness)
    
    cv2.line(image, (x + w, y + h), (x + w - corner_length, y + h), color, thickness)
    cv2.line(image, (x + w, y + h), (x + w, y + h - corner_length), color, thickness)

def process_images(images_path, output_path, face_classifier, max_images=300, corner_length=20):
    count = 0
    for image_name in os.listdir(images_path):
        image_path = os.path.join(images_path, image_name)
        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f'No se pudo leer la imagen {image_path}')
                continue

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            aux_frame = image.copy()

            faces = face_classifier.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                # Dibujar el recuadro con bordes en las esquinas en color celeste
                draw_corner_rect(image, x, y, w, h, (255, 255, 0), 2, corner_length)
                
                rostro = aux_frame[y:y + h, x:x + w]
                rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(os.path.join(output_path, f'rostro_{count}.jpg'), rostro)
                count += 1

                if count >= max_images:
                    print('Se ha alcanzado el límite de imágenes.')
                    return

            cv2.imshow('Image', image)
            k = cv2.waitKey(500)  # Espera 500 milisegundos entre imágenes

            if k == 27:  # Tecla ESC para salir
                print('Proceso interrumpido por el usuario.')
                break

        except Exception as e:
            print(f'Error procesando la imagen {image_name}: {e}')
    
    cv2.destroyAllWindows()

if __name__ == '__main__':
    personName = 'Cha Eun Woo'
    dataPath = 'Reconocimiento Facial/Data' 
    personPath = os.path.join(dataPath, personName)
    imagesPath = 'Cha Eun Woo'  # Ruta a la carpeta con las imágenes

    create_directory(personPath)
    
    faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    process_images(imagesPath, personPath, faceClassif)
