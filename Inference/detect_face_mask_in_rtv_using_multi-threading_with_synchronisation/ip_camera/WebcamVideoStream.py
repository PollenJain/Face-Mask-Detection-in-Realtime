import cv2
from threading import Thread
from timeit import default_timer as timer
from queue import Queue 
import csv

class WebcamVideoStream:
	def __init__(self, src=0):
		# initialize the video camera stream and read the first frame
		# from the stream
		self.stream = cv2.VideoCapture(src)
		(self.grabbed, self.frame) = self.stream.read()
		# initialize the variable used to indicate if the thread should
		# be stopped
		self.stopped = False
		# Using 2 fields for benchmarking
		self._numFrames = 0
		self._timeTaken = 0
		self._queue = Queue()
		

	def start(self):
		# start the thread to read frames from the video stream. Once, start method gets called from the driver function, the update method will be called by the run() of Thread class. update() gets executed in a separate thread.
		queue = Queue()
		Thread(target=self.update, args=()).start()
		return self
		
	def update(self):
		# keep looping infinitely until the thread is stopped
		with open("producer_thread_ip_camera.csv", 'w', newline='') as file:
			writer = csv.writer(file)
			writer.writerow(["Thread Frame #",  "Time spent in reading the frame (seconds) from web camera"])
			# loop over some frames
			while True:
				# if the thread indicator variable is set, stop the thread
				if self.stopped:
					return
				# otherwise, read the next frame from the stream
				start = timer()
				(self.grabbed, self.frame) = self.stream.read()
				end = timer()
				if self.grabbed:
					self._timeTaken = (end-start)
					self._numFrames += 1
					print("[INFO] Producer Thread : Time taken to read frame number", self._numFrames, "from ip camera is", self._timeTaken,"seconds")
					writer.writerow([self._numFrames, self._timeTaken])
					self._queue.put(self.frame.copy(), block=True, timeout=None) # Waits until a free slot is available

				
	
	def read(self):
		# return the frame most recently read
		#In order to access the most recently polled frame  from the stream , we’ll use the read method
		return self.frame
	def stop(self):
		# indicate that the thread should be stopped
		self.stopped = True
		self.stream.release()
		
	def readFrameNumber(self):
		return self._numFrames
		
	def timeTakenToReadAFrame(self):
		return self._timeTaken
		
	def readFromQueue(self):
		#Remove & Return only when the queue of frames is not empty. There is atleast one frame in it.
		if not self._queue.empty(): 
			return self._queue.get() #Remove and return an item from the queue.
		else:
			return None
			

		