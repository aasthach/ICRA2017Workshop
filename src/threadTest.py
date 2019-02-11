#Thread Testing
from helpers import sketchThread

import sys
import time

from PyQt5.QtCore import (QCoreApplication, QObject, QRunnable, QThread,
                          QThreadPool, pyqtSignal)



def stopFunc(sigStr):
	print("SigStr: " + sigStr)

if __name__ == '__main__':
	app = QCoreApplication([]); 
	objThread = QThread(); 
	ST = sketchThread(); 
	ST.moveToThread(objThread); 
	objThread.started.connect(ST.run); 
	objThread.finished.connect(app.exit); 
	ST.finished.connect(objThread.quit); 
	ST.signal.connect(stopFunc); 

	objThread.start(); 

	time.sleep(4); 
	ST.ending = True; 

	sys.exit(app.exec_()); 