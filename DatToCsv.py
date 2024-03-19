import sys
import pandas as pd

from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog, QLineEdit, QFileDialog, QPushButton, QLabel, \
    QMessageBox
from PyQt5.QtGui import QIcon
from PyQt5 import QtGui


class App(QWidget):

    def __init__(self):
        super().__init__()
        self.title = 'PyQt5 file dialogs - pythonspot.com'
        self.left = 10
        self.top = 10
        self.width = 300
        self.height = 250
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.move(300, 300);

        lab2 = QLabel(self)
        lab2.setText('Programma brutto per\nconvertire un log in csv')
        lab2.setFont(QtGui.QFont("Comic Sans", 15, QtGui.QFont.Black))
        lab2.move(30, 10)

        lab1 = QLabel(self)
        lab1.setText('© Luca Vecchio')
        lab1.setFont(QtGui.QFont("Arial", 10, QtGui.QFont.Black))
        lab1.move(100, 220)

        self.control_lab = QLabel(self)
        self.control_lab.setText('')
        self.control_lab.setFont(QtGui.QFont("Arial", 20, QtGui.QFont.Black))
        self.control_lab.move(60, 90)

        button = QPushButton('Convertimi un log', self)
        button.setToolTip('CONVERTILO!')
        button.move(110, 130)
        button.clicked.connect(self.on_click)

        button_group_function = QPushButton('Raggruppa i log', self)
        button_group_function.setToolTip('RAGGRUPPALI!')
        button_group_function.move(110, 180)
        button_group_function.clicked.connect(self.on_click_group_function)

        #self.openFileNameDialog()
        #self.openFileNamesDialog()
        #self.saveFileDialog()

        self.show()

    def openFileNameDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                  "All Files (*);;Python Files (*.py)", options=options)
        if fileName:
            #print(fileName)
            return fileName

    def openFileNamesDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        files, _ = QFileDialog.getOpenFileNames(self, "QFileDialog.getOpenFileNames()", "",
                                                "All Files (*);;Python Files (*.py)", options=options)
        if files:
            # print(files)
            return files

    def saveFileDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getSaveFileName(self, "QFileDialog.getSaveFileName()", "",
                                                  "All Files (*);;Text Files (*.txt)", options=options)
        if fileName:
            #print(fileName)
            return fileName

    @pyqtSlot()
    def on_click(self):
        print('clicked!!')
        name = self.openFileNameDialog()
        print('name: ' + name)
        saveName = self.saveFileDialog()
        print('saveName: ' + saveName)

        with open(name, 'r') as file:
            lines = file.readlines()
            setting_phase = False
            data_phase = False
            dic_header = {}
            dic_content = {}

            for line in lines:
                #print(line)
                if 'setting:' in line:
                    setting_phase = True
                    continue
                if 'data:' in line:
                    data_phase = True
                    setting_phase = False
                    dic_content = {f: [] for f in dic_header.keys()}
                    print(dic_content)
                    print(dic_header)
                    continue

                if setting_phase == True:
                    fields = line.split(',')
                    headers = []
                    headers.append('time')
                    headers.append('lap')
                    for field in fields[1:]:
                        if field[0] == '{':
                            headers.append(field[6:])

                    dic_header[fields[0]] = ','.join(headers)

                if data_phase == True:
                    values = line.split(',')
                    try:
                        dic_content[values[0]].append(','.join(values[1:]))
                    except KeyError:
                        pass

        for k in dic_header.keys():
            file_name = saveName + '_' + k + '.csv'
            with open(file_name, 'w+') as file:
                file.write(dic_header[k] + '\n')
                for line in dic_content[k]:
                    file.write(line)

            print(file_name, 'Done.')


        dlg = QMessageBox(self)
        dlg.setWindowTitle("Errore")
        dlg.setText("Processo fallito correttamente.\nScherzone, ha funzionato alla grande")
        button = dlg.exec()

    @pyqtSlot()
    def on_click_group_function(self):
        names = self.openFileNamesDialog()
        print('names: ' + str(names))
        saveName = self.saveFileDialog()
        print('saveName: ' + saveName)

        df = pd.DataFrame()

        for name in names:
            df_name = pd.read_csv(name)
            df = pd.concat([df, df_name])
        
        df['time'] = pd.to_datetime(df['time'], format='%H:%M:%S.%f')
        df.sort_values(by=['time'], inplace=True)
        df.fillna(method='bfill', inplace=True)
        df.fillna(method='ffill', inplace=True)
        df['time'] = df['time'].dt.strftime('%H:%M:%S.%f')

        print(df.head(10))
        df.to_csv(saveName + '_grouped.csv', index=False)

        dlg = QMessageBox(self)
        dlg.setWindowTitle("Ha funzionato!")
        dlg.setText("Incredibile fra\ngoditi il csv raggruppato")
        button = dlg.exec()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
