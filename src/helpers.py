import time

#########################################################
#Found at: 
#https://stackoverflow.com/questions/5849800/tic-toc-functions-analog-in-python
#########################################################

class Timer(object):
	def __init__(self,name):
		self.name = name; 

	def __enter__(self):
		self.tstart = time.time(); 

	def __exit__(self,type,value,traceback):
		if(self.name):
			print('[%s]' % self.name),
		print('Elapsed: %s' % (time.time()-self.tstart)); 



#########################################################
#Found at: 
#https://stackoverflow.com/questions/6783194/background-thread-with-qthread-in-pyqt
#########################################################


from PyQt5.QtCore import QThread, QObject
import time; 
import PyQt5.QtCore as QtCore

class sketchThread(QObject):

	signal = QtCore.pyqtSignal(str); 
	finished = QtCore.pyqtSignal(); 

	def __init__(self,imageScene=None):
		#set up pointer to sketch plane
		self.view = imageScene; 
		self.ending = False
		QObject.__init__(self); 

	def run(self):
		count = 0; 
		while not self.ending:
			time.sleep(1); 
			self.signal.emit("{}".format(count)); 
			print(count); 
			count += 1; 
		self.finished.emit(); 

