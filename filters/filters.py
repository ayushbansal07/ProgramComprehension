import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
class Filters :
	def __init__(self,base_dir):
		self.GAUSSIAN_KERNEL = np.array([5,15,60,15,5])/100
		self.LAPLACIAN_FILTER = np.array([-1,-1,4,-1,-1])
		self.FIRST_DERIVATIVE = np.array([-1,-2,0,2,1])
		self.BASE_DIR = base_dir

	def readfile(self,filename):
		return pd.read_excel(self.BASE_DIR + filename)

	def filter_data(self,data):
		filtered_data = np.array(data)
		fil_data = []
		for x in filtered_data:
			if x[1] != 0:
				fil_data.append((x[1],x[-1]))
		filtered_data = np.array(fil_data)
		return filtered_data

	def read_and_preprocess_data(self,filename):
		return self.filter_data(self.readfile(filename))

	def plot_data(self,data,saveplace=None,isDerivative = False):
		plt.figure(figsize=(20,12))
		plt.scatter(data[:,1],data[:,0])
		y_min = np.min(data[:,0])
		y_max = np.max(data[:,0])
		if isDerivative:
			thresh_lim = -60
		else:
			thresh_lim = 0
		plt.ylim(max(60,y_max+5),min(thresh_lim,y_min-5))
		if saveplace is not None:
			plt.savefig(saveplace)
			plt.close()
		else:
			plt.show()

	def apply_filter(self,data,kernel,kernel_len):
		pd = int(kernel_len/2)
		ln = len(data)
		newdata = []
		for i in range(pd):
			newdata.append(data[i])
		for i in range(pd,ln - pd):
			val = np.sum(data[i-pd:i+pd+1,0]*kernel)
			newdata.append((val,data[i,1]))
		for i in range(ln-pd,ln) :
			newdata.append(data[i])
		return np.array(newdata)

	def conservative_smoothing(self,data,kernel_len):
		pd = int(kernel_len/2)
		ln = len(data)
		newdata = []
		for i in range(pd):
			newdata.append(data[i])
		for i in range(pd,ln - pd):
			#val = np.sum(data[i-pd:i+pd+1,0]*kernel)
			neighbors = data[i-pd:i,0]
			neighbors = np.concatenate((neighbors,data[i+1:i+1+pd,0]))
			if data[i,0] < np.min(neighbors):
				newdata.append((np.min(neighbors),data[i,1]))
			elif data[i,0] > np.max(neighbors):
				newdata.append((np.max(neighbors),data[i,1]))
			else:
				newdata.append((data[i,0],data[i,1]))
		for i in range(ln-pd,ln) :
			newdata.append(data[i])
		return np.array(newdata)

	def apply_thresh(self,data,th,keep=True):
		res = []
		for x in data:
			if abs(x[0])<th:
				if keep:
					res.append((0,x[1]))
			else:
				res.append((x[0],x[1]))
		return np.array(res)

	def apply_range_thresh(self,data,range_min,range_max,keep=True):
		res = []
		for x in data:
			if abs(x[0]) >= range_min and abs(x[0]) <= range_max:
				res.append((x[0],x[1]))
			else:
				if keep:
					res.append((0,x[1]))

		return np.array(res)

	def seperate_data_range(self,data,first_der,range_min,range_max):
		a = []
		b = []
		for i,x in enumerate(data):
			if abs(first_der[i][0]) >= range_min and abs(first_der[i][0]) <= range_max:
				b.append((x[0],x[1]))
			else:
				a.append((x[0],x[1]))

		return np.array(a), np.array(b)

	def plot_colored_data(self,data_blue,data_red,saveplace=None):
		plt.figure(figsize=(20,12))
		plt.scatter(data_blue[:,1],data_blue[:,0],c='b')
		max_y = np.max(data_blue[:,0])
		if data_red.shape[0] > 0:
		    plt.scatter(data_red[:,1],data_red[:,0],c='r')
		    max_y = max(max_y,np.max(data_red[:,0]))
		plt.ylim(max(60,max_y+10),0)
		if saveplace is not None:
		    plt.savefig(saveplace)
		    plt.clf()
		else:
		    plt.show()
