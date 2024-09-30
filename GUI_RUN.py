# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 16:00:37 2024

@author: lenovo
"""

import json
import os
import sys
import subprocess
import traceback
from PyQt5 import QtCore, QtWidgets, QtGui
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(1289, 736)
        self.originalImage = QtWidgets.QGraphicsView(Dialog)
        self.originalImage.setGeometry(QtCore.QRect(320, 60, 281, 221))
        self.originalImage.setObjectName("originalImage")
        self.segmentedImage = QtWidgets.QGraphicsView(Dialog)
        self.segmentedImage.setGeometry(QtCore.QRect(630, 60, 281, 221))
        self.segmentedImage.setObjectName("segmentedImage")
        self.clearSelection = QtWidgets.QPushButton(Dialog)
        self.clearSelection.setGeometry(QtCore.QRect(180, 230, 111, 41))
        self.clearSelection.setObjectName("clearSelection")
        self.textBrowser = QtWidgets.QTextBrowser(Dialog)
        self.textBrowser.setGeometry(QtCore.QRect(20, 60, 271, 141))
        self.textBrowser.setObjectName("textBrowser")
        self.textBrowser_2 = QtWidgets.QTextBrowser(Dialog)
        self.textBrowser_2.setGeometry(QtCore.QRect(320, 20, 281, 31))
        self.textBrowser_2.setObjectName("textBrowser_2")
        self.textBrowser_3 = QtWidgets.QTextBrowser(Dialog)
        self.textBrowser_3.setGeometry(QtCore.QRect(630, 20, 281, 31))
        self.textBrowser_3.setObjectName("textBrowser_3")
        self.line = QtWidgets.QFrame(Dialog)
        self.line.setGeometry(QtCore.QRect(30, 290, 1231, 21))
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.embeddingsView = QtWidgets.QGraphicsView(Dialog)
        self.embeddingsView.setGeometry(QtCore.QRect(320, 320, 591, 391))
        self.embeddingsView.setObjectName("embeddingsView")
        self.chooseEmbeddingDirs = QtWidgets.QPushButton(Dialog)
        self.chooseEmbeddingDirs.setGeometry(QtCore.QRect(30, 600, 271, 41))
        self.chooseEmbeddingDirs.setObjectName("chooseEmbeddingDirs")
        self.textBrowser_4 = QtWidgets.QTextBrowser(Dialog)
        self.textBrowser_4.setGeometry(QtCore.QRect(30, 320, 271, 192))
        self.textBrowser_4.setObjectName("textBrowser_4")
        self.chooseInputImage = QtWidgets.QPushButton(Dialog)
        self.chooseInputImage.setGeometry(QtCore.QRect(20, 230, 151, 41))
        self.chooseInputImage.setObjectName("chooseInputImage")
        self.Makeembeddingvector = QtWidgets.QPushButton(Dialog)
        self.Makeembeddingvector.setGeometry(QtCore.QRect(30, 540, 271, 41))
        self.Makeembeddingvector.setObjectName("Makeembeddingvector")
        self.textBrowser_5 = QtWidgets.QTextBrowser(Dialog)
        self.textBrowser_5.setGeometry(QtCore.QRect(940, 20, 281, 31))
        self.textBrowser_5.setObjectName("textBrowser_5")
        self.textBrowser_6 = QtWidgets.QTextBrowser(Dialog)
        self.textBrowser_6.setGeometry(QtCore.QRect(940, 60, 281, 221))
        self.textBrowser_6.setObjectName("textBrowser_6")
        self.textBrowser_7 = QtWidgets.QTextBrowser(Dialog)
        self.textBrowser_7.setGeometry(QtCore.QRect(940, 320, 281, 391))
        self.textBrowser_7.setObjectName("textBrowser_7")
        self.newdomain = QtWidgets.QPushButton(Dialog)
        self.newdomain.setGeometry(QtCore.QRect(30, 660, 271, 41))
        self.newdomain.setObjectName("newdomain")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)
       
        # Connect buttons to functions
        self.chooseInputImage.clicked.connect(self.run_segmentation_script)
        self.clearSelection.clicked.connect(self.clear_selection)
        self.Makeembeddingvector.clicked.connect(self.run_text_script)
        self.chooseEmbeddingDirs.clicked.connect(self.run_embedding_visu_script)
        self.newdomain.clicked.connect(self.run_newdomain_script)

        self.canvas = None  # This will hold the matplotlib canvas

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.clearSelection.setText(_translate("Dialog", "Clear Selection"))
        
        # Load all text content from JSON file
        text_content = self.load_text_from_json("text_content.json")
        
        # Set the text from the JSON file into the respective text browsers        
        self.textBrowser.setHtml(text_content.get("text1", ""))
        self.textBrowser_2.setHtml(text_content.get("text2", ""))
        self.textBrowser_3.setHtml(text_content.get("text3", ""))
        self.textBrowser_4.setHtml(text_content.get("text4", ""))
        self.textBrowser_5.setHtml(text_content.get("text5", ""))

        
        self.chooseEmbeddingDirs.setText(_translate("Dialog", "Choose Domains"))
        self.chooseInputImage.setText(_translate("Dialog", "Choose Input Image"))
        self.Makeembeddingvector.setText(_translate("Dialog", "Make Embedding Vector"))
        self.newdomain.setText(_translate("Dialog", "New Domain"))
        

    def run_segmentation_script(self):
        try:
            # Get the current directory of this script
            current_directory = os.path.dirname(os.path.abspath(__file__))

            # Construct the full path to segmentation.py
            script_path = os.path.join(current_directory, 'Segmentation.py')

            # Run the segmentation.py script and capture its output
            result = subprocess.run(["python", script_path], capture_output=True, text=True)

            # Get the output from the segmentation script
            output = result.stdout.strip()

            if result.returncode == 0:
                # The output will be in the format "input_image_path,segment_image_path"
                input_image_path, segment_image_path = output.split(',')

                # Display images in the QGraphicsViews
                self.display_image(self.originalImage, input_image_path)
                self.display_image(self.segmentedImage, segment_image_path)
            else:
                print(f"Segmentation script failed: {result.stderr}")
        except Exception as e:
            print(f"Error running segmentation script: {e}")

    def display_image(self, graphics_view, image_path):
        # Load the image into a QPixmap
        pixmap = QtGui.QPixmap(image_path)
        if pixmap.isNull():
            print(f"Failed to load image: {image_path}")
            return
        
        # Create a scene to display the pixmap in the QGraphicsView
        scene = QtWidgets.QGraphicsScene()
        scene.addPixmap(pixmap)
        graphics_view.setScene(scene)
        graphics_view.fitInView(scene.sceneRect(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)

    def clear_selection(self):
        # Clear the graphics views when "Clear Selection" is clicked
        self.originalImage.setScene(None)
        self.segmentedImage.setScene(None)
        self.textBrowser_6.clear()

    # Helper function to load text from a JSON file
    def load_text_from_json(self, filename):
        try:
            # Get the directory of the current script
            script_dir = os.path.dirname(os.path.abspath(__file__))

            # Construct the full path to the JSON file
            file_path = os.path.join(script_dir, filename)
            
            # Load and return the JSON data
            with open(file_path, 'r') as file:
                return json.load(file)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading JSON file {filename}: {e}")
            return {}

    def run_embedding_visu_script(self):
        try:
            current_directory = os.path.dirname(os.path.abspath(__file__))
            script_path = os.path.join(current_directory, 'EmbeddingVisu.py')

            # Clear the previous plot if exists
            if self.canvas:
                self.embeddingsView.scene().clear()

            # Import and call EmbeddingVisu.main, which now returns the figure
            import EmbeddingVisu
            fig = EmbeddingVisu.main()

            # Create a matplotlib canvas and add it to the embeddingsView
            self.canvas = FigureCanvas(fig)
            scene = QtWidgets.QGraphicsScene(self.embeddingsView)
            scene.addWidget(self.canvas)
            self.embeddingsView.setScene(scene)

        except Exception as e:
            print(f"Error running EmbeddingVisu script: {e}")
            


    def run_text_script(self):
        try:
            # Call text.py and capture its output
            result = subprocess.run(['python', 'text.py'], check=True, text=True, capture_output=True)

            # Directly display the standard output of text.py in textBrowser_6
            self.textBrowser_6.setPlainText(result.stdout)

            # Output error messages (if any)
            if result.stderr:
                self.textBrowser_6.append(f"Error output:\n{result.stderr}")

        except subprocess.CalledProcessError as e:
            self.textBrowser_6.setPlainText(f"Subprocess error: {e.stderr}")
        except Exception as e:
            self.textBrowser_6.setPlainText(f"An error occurred: {str(e)}")



    def run_newdomain_script(self):
        try:
            # Call newdomain.py and capture its output
            result = subprocess.run(['python', 'newdomain.py'], check=True, text=True, capture_output=True)

            # Directly display the standard output of newdomain.py in textBrowser_7
            self.textBrowser_7.setPlainText(result.stdout)

            # Output error messages (if any)
            if result.stderr:
                self.textBrowser_7.append(f"Error output:\n{result.stderr}")

        except subprocess.CalledProcessError as e:
            self.textBrowser_7.setPlainText(f"Subprocess error: {e.stderr}")
        except Exception as e:
            self.textBrowser_7.setPlainText(f"An error occurred: {str(e)}")



if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    app.exec()