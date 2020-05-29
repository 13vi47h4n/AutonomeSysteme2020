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
