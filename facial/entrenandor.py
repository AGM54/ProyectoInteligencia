import cv2
import os
import numpy as np

def get_people_list(data_path):
    return os.listdir(data_path)

def read_images(data_path, people_list):
    labels = []
    faces_data = []
    label = 0

    for name_dir in people_list:
        person_path = os.path.join(data_path, name_dir)

        for file_name in os.listdir(person_path):
            image_path = os.path.join(person_path, file_name)
            image = cv2.imread(image_path, 0)

            if image is not None:
                labels.append(label)
                faces_data.append(image)
            else:
                print(f'Error leyendo la imagen {image_path}')
        
        label += 1

    return faces_data, labels

def train_model(faces_data, labels):
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(faces_data, np.array(labels))
    return face_recognizer

def save_model(face_recognizer, file_name):
    face_recognizer.write(file_name)

def main():
    data_path = 'Reconocimiento Facial/Data'
    model_file = 'modeloentrenado.xml'

    people_list = get_people_list(data_path)
    print('Lista de personas:')
    for person in people_list:
        print(person)

    faces_data, labels = read_images(data_path, people_list)
    face_recognizer = train_model(faces_data, labels)
    save_model(face_recognizer, model_file)

if __name__ == '__main__':
    main()
