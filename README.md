# HandGestureRec
Repositorio para el proyecto Hand Gesture Recognition, este contiene los siguientes archivos:

## OrderDatabase.m
Este script ordena la base de datos Leap Motion Dynamic Hand Gesture (LMDHG) que se utiliza para el entrenamiento de una red neuronal artificial. Esta corresponde a la lectura de 46 sensores que dan la orientación espacial de las manos para clasificarlo en 13 gestos diferentes.

## MLP.py
Este script realiza una red neuronal profunda que se puede generalizar para cualquier número de capas y neuronas deseadas.

## LecturaSensores.ino
Este script sirve para la lectura de los sensores de flexión capacitivo, resistivo y las dos IMUs.

## SensoresCSVfile.pde
Este script corresponde al código ejecutable en el software processing para recibir datos del monitor serie del Arduino y convertirlos en un archivo CSV. Este script se utiliza para la realización de la base de datos.

## DatabaseHGR.csv
Este archivo corresponde a un ejemplo de lo obtenido del script de processing para generar archivos CSV con las lecturas de los sensores asignadas y la clase asignada.
