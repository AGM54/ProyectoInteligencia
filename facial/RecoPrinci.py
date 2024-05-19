import cv2
import os

def load_model(model_path):
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.read(model_path)
    return face_recognizer

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

def detect_faces(image, face_classifier):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    return faces, gray

def recognize_faces(image, faces, gray, face_recognizer, labels):
    for (x, y, w, h) in faces:
        rostro = gray[y:y + h, x:x + w]
        rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
        result = face_recognizer.predict(rostro)

        confidence = result[1]
        label = labels[result[0]] if confidence < 70 else 'Desconocido'
        color = (0, 0, 255) if label != 'Desconocido' else (0, 0, 255)
        draw_corner_rect(image, x, y, w, h, (255, 255, 0), 2, 20)
        
        # Mostrar etiqueta y confianza en negro
        cv2.putText(image, f'{label}', (x, y - 25), 1, 1.3, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(image, f'Conf: {confidence:.2f}', (x, y - 5), 1, 1.3, (255, 255, 255), 1, cv2.LINE_AA)
        
    return image

def process_test_images(test_images_path, face_classifier, face_recognizer, labels):
    window_name = 'Reconocimiento Facial'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    for image_name in os.listdir(test_images_path):
        image_path = os.path.join(test_images_path, image_name)
        image = cv2.imread(image_path)
        if image is None:
            continue
        
        faces, gray = detect_faces(image, face_classifier)
        image = recognize_faces(image, faces, gray, face_recognizer, labels)
        
        cv2.imshow(window_name, image)
        cv2.resizeWindow(window_name, image.shape[1], image.shape[0])

        k = cv2.waitKey(1000)  # Espera 1 segundo entre imágenes

        if k == 27:  # Tecla ESC para salir
            break

    cv2.destroyAllWindows()

def main():
    data_path = 'Reconocimiento Facial/Data'
    model_path = 'modeloentrenado.xml'
    test_images_path = 'no shakira'

    labels = os.listdir(data_path)
    print('Lista de personas:')
    for person in labels:
        print(person)

    face_recognizer = load_model(model_path)
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    process_test_images(test_images_path, face_classifier, face_recognizer, labels)

if __name__ == '__main__':
    main()
