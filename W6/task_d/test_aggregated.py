import csv
import numpy as np

# Diccionario para almacenar las sumas de probabilidades por nombre de video
video_probabilities = {}

video_truth = {}
# Función para sumar las probabilidades y almacenarlas en el diccionario
def sum_probabilities(video_name, probabilities,truth):
    if video_name in video_probabilities:
        video_probabilities[video_name] = [x + y for x, y in zip(video_probabilities[video_name], probabilities)]
    else:
        video_probabilities[video_name] = probabilities
        video_truth[video_name]=truth

# Ruta al archivo CSV
csv_file = 'predictions_test_set.csv'

# Abrir el archivo CSV y procesar cada línea
with open(csv_file, 'r') as file:
    csv_reader = csv.reader(file)
    next(csv_reader)  # Saltar la primera línea (encabezado)
    for row in csv_reader:
        probabilities=[]
        video = row[0].split('.')
        video_name = video[0]
        print(row[3])
        probabilities.append(eval(row[3][1:]))
        probabilities.append(eval(row[4]))
        probabilities.append(eval(row[5]))
        probabilities.append(eval(row[6]))
        probabilities.append(eval(row[7]))
        probabilities.append(eval(row[8]))
        probabilities.append(eval(row[9][:-1]))
        gt=row[1]
        #print(probabilities)  # Convertir la cadena de probabilidades a una lista de Python
        sum_probabilities(video_name, probabilities,gt)

# Imprimir las sumas de probabilidades por nombre de video
acc_count=0
for video_name, sum_prob in video_probabilities.items():
    #print(f"Suma de probabilidades para '{video_name}': {sum_prob}")

    if (np.argmax(sum_prob)+1) == int(video_truth[video_name]):
        print(np.argmax(sum_prob)+1)
        print(video_truth[video_name])
        acc_count +=1
acc_final= acc_count/len(video_truth)

print(f"New test accuracy: '{acc_final}'")