# Autonome Systeme 2020
Team IW276SS20P2 Affective Computing - Face Expression Recognition

# Dateien
pipeline_main.py
- Dies ist das Hauptskript in der alle Pipeline Segmente zusammengefügt sind.

text_export.py
- Export der Logging Informationen in eine yml-Datei

convert2trt.py
- Das Skript konvertiert unter Angabe des Modellnamens, den Quellpfad und Zielpfad ein ResNet Modell in ein TorchTRT Modell um und speichert es im Zielpfad ab. 

requirements.txt:
- Auflistung aller direkter Abhängigkeiten

face_expression_recognition.py
- Das Skript für die Gesichtsausdruckserkennung. Das TorchTRT Modell wird geladen und das einzelne Input Bild auf die passende Größe skaliert. Das Modell gibt die erkannte Emotion zurück.

# Installation

## Grobunterteilung
1. Dependencies installieren
2. torch-Modell konvertieren

### Dependencies installieren
Installation in virtual environment:
- Pytorch
    ```
    wget https://nvidia.box.com/shared/static/3ibazbiwtkl181n95n9em3wtrca7tdzp.whl -O torch-1.5.0-cp36-cp36m-linux_aarch64.whl
    sudo apt-get install python3-pip libopenblas-base
    pip install Cython
    pip install numpy torch-1.4.0-cp36-cp36m-linux_aarch64.whl
    ```
- Torchvision
    ```
    sudo apt-get install libjpeg-dev zlib1g-dev
    git clone --branch v0.6.0 https://github.com/pytorch/vision torchvision
    cd torchvision
    sudo python setup.py install
    cd ../
    ```
- torch2trt
    Anleitung unter https://github.com/NVIDIA-AI-IOT/torch2trt#setup folgen
- Face_Recognition (https://github.com/ageitgey/face_recognition)
    ```
    pip install face_recognition
    ```

### Torch Modell konvertieren
- Konvertierung
    ```
    python convert2trt.py <mode> <source> <target>
    ```
- Info: die Pipeline erwartet das kovertierte Modell unter `'./models/resnet50.224.trt.pth'`. Bei Bedarf in `face_expression_recognition.py` anpassen.

# Benutzung
Die Pipeline kann mit 
```bash
python pipeline_main.py <source>
```
gestartet werden. Als source sind u.a.
- Kamera-Index (z.B. 0)
- IP-Kamera (als URL)
- Video (Pfad, z.B. mp4)

möglich.

In der Datei config.yml können Einstellungen geändert werden, um Genauigkeit vs Geschwindigkeit an die Quelle anpassen zu können.