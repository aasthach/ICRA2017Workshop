
import multiprocessing

def spawn():
	print("test!"); 


if __name__ == '__main__':
	for i in range(0,5):
		p = multiprocessing.Process(target=spawn); 
		p.start(); 