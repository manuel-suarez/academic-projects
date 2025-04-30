import tensorflow as tf
import timeit
import pandas as pd
from datetime import timedelta
import matplotlib.pyplot as plt


#https://medium.com/@upu1994/how-easy-is-making-custom-keras-callbacks-c771091602da
#https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/Callback
class History(tf.keras.callbacks.Callback):

	def __init__(self,accuracy_filename, loss_filename,
		checkOn, metric="accuracy"):

		self.count = 0
		self.accuracy_filename = accuracy_filename
		self.loss_filename = loss_filename
		self.checkOn = checkOn
		self.metric = metric
		self.columns = ["train_" + self.metric, "cross_val_" + self.metric, "train_loss", "cross_val_loss", "time"]


	def setNamePlots(self, accuracy_filename, loss_filename):
		self.accuracy_filename = accuracy_filename
		self.loss_filename = loss_filename

	def saveCSV(self):

		print("dataframe ", len(self.data))

		output_file = self.accuracy_filename + ".csv"
		out_dF = pd.DataFrame(self.data, columns = self.columns)
		out_dF.to_csv(output_file, index=None)


	def savePlot(self):

		df = pd.DataFrame(self.data, columns = self.columns)

		
		plt.plot( df[self.columns[0]] )
		plt.plot( df[self.columns[1]] )

		plt.title('model accuracy')
		plt.ylabel('accuracy')
		plt.xlabel('epoch')
		plt.legend(['train', 'validation'], loc='upper left')
		plt.savefig( "%s_%04d.png" % (self.accuracy_filename , self.count ) )
		plt.clf()

		# summarize history for loss
		plt.plot( df[self.columns[2]] )
		plt.plot( df[self.columns[3]] )
		
		plt.title('model loss')
		plt.ylabel('loss')
		plt.xlabel('epoch')
		plt.legend(['train', 'validation'], loc='upper left')
		plt.savefig( "%s_%04d.png" % (self.loss_filename , self.count ) )

		plt.clf()

	def on_train_begin(self, logs={}):

		self.acc = []
		self.val_acc = []
		self.loss = []
		self.val_loss = []
		self.data = []

	def on_epoch_begin(self, epoch, logs={}):

		self.start_time = timeit.default_timer()

	def on_epoch_end(self, epoch, logs=None):

		end_time = timeit.default_timer()

		#print( logs )

		metric =  logs.get(self.metric)
		val_metric = logs.get("val_" + self.metric)

		loss = logs.get("loss")
		val_loss = logs.get("val_loss")

		self.acc.append( metric )
		self.val_acc.append( val_metric )
		self.loss.append( loss )
		self.val_loss.append( val_loss )

		
		stime = str(timedelta(seconds=end_time - self.start_time))

		self.data.append( [self.acc[-1], self.val_acc[-1], self.loss[-1], self.val_loss[-1], stime] )

		if ( self.count!=0 and self.count % self.checkOn == 0):

			print ("... epoch : ", self.count ," acc: ", metric , "val_" + self.metric + ":", val_metric )
			self.savePlot()
			self.saveCSV()
			

		self.count += 1
