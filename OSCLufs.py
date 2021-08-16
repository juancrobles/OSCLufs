#OSCLufs
#Office Hours Global Community Project
#Created and maintained by Andy Carluccio - Washington, D.C.

#Contributors:
#Juan C. Robles - Mexico City, MX

#Last updated 8/11/2021

#OSC variables & libraries
from app_setup import CHANNELS, FRAMES_PER_BUFFER, SAMPLE_RATE
from pythonosc import dispatcher
from pythonosc import osc_server
from pythonosc import osc_message_builder
from pythonosc import udp_client 

#Argument management and system
import argparse
import sys
import time
from enum import Enum

#Numpy Library
import numpy as np

#Loudness Processing Library
import pyloudnorm as pyln

#Files (No longer needed, but left for future use)
import wave
import soundfile

#Print the names of all available audio devices
import pyaudio

#App setup
from app_setup import (
    SAMPLE_RATE,
	FRAMES_PER_BUFFER,
	CHANNELS,
	MIN_LOUDNESS,
	DURATION)

class DeviceTypes(Enum):
	ALL_DEVICES = 0
	INPUT_DEVICES = 1
	OUTPUT_DEVICES = 2

class AudioManager(object):
	def __init__(self):
		super().__init__()
		self._pa = pyaudio.PyAudio()

	def getDevicesCount(self, type=DeviceTypes.ALL_DEVICES):
		# print("get devices count")
		count = 0

		if(type == DeviceTypes.ALL_DEVICES):
			count = self._pa.get_device_count()
		else:
			for i in range(self._pa.get_device_count()):
				device = self._pa.get_device_info_by_index(i)

				if(type == DeviceTypes.INPUT_DEVICES and device.get("maxInputChannels") > 0):
					count += 1
				elif(type == DeviceTypes.OUTPUT_DEVICES and device.get("maxOutputChannels") > 0):
					count += 1

		return count
 
	def getDevices(self, type=DeviceTypes.ALL_DEVICES):
		# print("get devices")
		devices = []

		for i in range(self._pa.get_device_count()):
			# print("device:", self._pa.get_device_info_by_index(i))
			# print()
			device = self._pa.get_device_info_by_index(i)

			if(type == DeviceTypes.ALL_DEVICES):
				devices.append(device.get("name"))
			elif(type == DeviceTypes.INPUT_DEVICES and device.get("maxInputChannels") > 0):
				devices.append(device.get("name"))
			elif(type == DeviceTypes.OUTPUT_DEVICES and device.get("maxOutputChannels") > 0):
				devices.append(device.get("name"))

		return devices

# JCR: Resources - https://www.youtube.com/watch?v=at2NppqIZok and https://github.com/aniawsz/rtmonoaudio2midi
class StreamProcessor(object):
	def __init__(self, input_device, \
				format = pyaudio.paFloat32, \
				input = True, \
				sample_rate = SAMPLE_RATE, \
				frames_per_buffer = FRAMES_PER_BUFFER, \
				channels = CHANNELS):
		self._input_device = input_device
		self._format = format
		self._input = input
		self._sample_rate = sample_rate
		self._frames_per_buffer = frames_per_buffer
		self._channels = channels

	def run(self):
		pya = pyaudio.PyAudio()
		self._stream = pya.open(
			format=self._format,
			channels=self._channels,
			rate=self._sample_rate,
			input=self._input,
			frames_per_buffer=self._frames_per_buffer,
			stream_callback=self._process_frame
		)
		self._stream.start_stream()

		while self._stream.is_active(): #and not self._stream.raw_input():
			time.sleep(0.1)

		self._stream.stop_stream()
		self._stream.close()
		pya.terminate()

	def _process_frame(self, data, frame_count, time_info, status_flag):
		self._data = data
		return (data, pyaudio.paComplete)

	def getData(self):
		data = self._data
		self._data = None
		#print(data)
		return data

audio_stream = pyaudio.PyAudio()

print("ALL SYSTEM AUDIO DEVICES:")
print()

for i in range(audio_stream.get_device_count()):
	sys.stdout.write(str(i))
	sys.stdout.write(" ")
	sys.stdout.write(audio_stream.get_device_info_by_index(i).get("name"))
	sys.stdout.write('\n')

print()
print("Please select the audio device to listen to:")

micIndex = int(input())

def getLufs(unused_addr):
	# print("Running getLufs function!")
	# Initialize local variables
	loudness = -70.0
	frames = []

	# Initialize meter
	meter = pyln.Meter(SAMPLE_RATE)

	# Initialize Audio capture
	sp = StreamProcessor(micIndex)

	# Capture audio frames for duration
	for i in range(0, int(SAMPLE_RATE / FRAMES_PER_BUFFER * DURATION)):
		sp.run()
		data = sp.getData()
		frames.append(data)

	# Concatenate frames
	total_data = b''.join(frames)
	data_samples = np.frombuffer(total_data,dtype=np.float32)

	# print("total data count:", len(total_data))
	# print("data samples count", len(data_samples))

	# Calculate loudness
	inmediate_loudness = meter.integrated_loudness(data_samples) # measure loudness

	# Limiter lower output value
	if(inmediate_loudness < MIN_LOUDNESS):
		loudness = MIN_LOUDNESS
	else:
		loudness = inmediate_loudness

	#send the loundess as OSC
	client.send_message("/OSCLufs/lufs", loudness)

def getAudioDevices(unused_addr, args):
	client.send_message("/OSCLufs/audio/devices", args[0])

def getAudioDevicesCount(unused_addr, args):
	client.send_message("/OSCLufs/audio/devicesCount", args[0])

if __name__ == "__main__":
	#Initialize audio manager
	am = AudioManager()

	#Get the networking info from the user
	print("Would you like to [1] Input network parameters or [2] use default: 127.0.0.1:1234 (send) and 127.0.0.1:7070 (receive)?")
	print("Enter 1 or 2")
	
	send_ip = "127.0.0.1"
	send_port = 1234
	in_port = 7070

	selection = int(input())
	if(selection == 1):
		print("Input network parameters")
		send_ip = str(input("Send ip?: "))
		send_port = int(input("Send port?: "))
		in_port = int(input("Receive port?: "))
	else:
		print("Using default network settings")
	
	#sending osc messages on
	client = udp_client.SimpleUDPClient(send_ip,send_port)
	sys.stdout.write("Opened Client on: ")
	sys.stdout.write(send_ip)
	sys.stdout.write(":")
	sys.stdout.write(str(send_port))
	sys.stdout.write('\n')

	#catches OSC messages
	dispatcher = dispatcher.Dispatcher()
	dispatcher.map("/OSCLufs/getLufs", getLufs)
	#audio commands
	dispatcher.map("/OSCLufs/audio/getDevices", getAudioDevices, am.getDevices(DeviceTypes.INPUT_DEVICES))
	dispatcher.map("/OSCLufs/audio/getDevicesCount", getAudioDevicesCount, am.getDevicesCount(DeviceTypes.INPUT_DEVICES))
	
	#set up server to listen for osc messages
	server = osc_server.ThreadingOSCUDPServer((send_ip,in_port),dispatcher)
	print("Starting Server on {}".format(server.server_address))
	
	#Print API
	print("OSC Networking Established")
	print()
	print("OSC API for Controlling OSCLufs:")
	print("/OSCLufs/getLufs: Request a lufs reply")
	print()
	print("OSC API for Receiving Text from OSCLufs:")
	print("/OSCLufs/lufs {float num}: The lufs")
	print()

	#Print Audio API
	print("OSC Audio API")
	print()
	print("OSC API for Getting audio input devices count:")
	print("/OSCLufs/audio/getDevicesCount: Request input devices count")
	print()
	print("OSC API for Receiving Text from OSCLufs:")
	print("/OSCLufs/audio/devicesCount {integer num}: Input devices count")
	print()
	print("OSC API for Getting audio input devices:")
	print("/OSCLufs/audio/getDevices: Request input devices")
	print()
	print("OSC API for Receiving Text from OSCLufs:")
	print("/OSCLufs/audio/devices {List string}: Input devices name list")
	print()
	#begin the infinite loop
	server.serve_forever()
