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
# import pyloudnorm as pyln
from pyebur128 import (
    ChannelType, MeasurementMode, R128State, get_loudness_shortterm
)

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

class DeviceType(Enum):
	INPUT_OUTPUT_DEVICE = 0
	INPUT_DEVICE = 1
	OUTPUT_DEVICE = 2

class AudioDevice(object):
	def __init__(self, device) -> None:
		super().__init__()
		self._device = device

	def index(self) -> int: 
		return int(self._device.get("index"))

	def name(self) -> str:
		return str(self._device.get("name"))

	def sampleRate(self) -> int: 
		return int(self._device.get("defaultSampleRate"))

	def type(self) -> DeviceType: 
		if(self._device.get("maxInputChannels") != 0 and self._device.get("maxOutputChannels") == 0):
			return DeviceType.INPUT_DEVICE
		elif(self._device.get("maxInputChannels") == 0 and self._device.get("maxOutputChannels") != 0):
			return DeviceType.OUTPUT_DEVICE

		return DeviceType.INPUT_OUTPUT_DEVICE

	def channels(self, type: DeviceType):
		if(type == DeviceType.INPUT_DEVICE):
			return self._device.get("maxInputChannels")
		elif(type == DeviceType.OUTPUT_DEVICE):
			return self._device.get("maxOutputChannels")
		else: self._device.get("maxInputChannels") + self._device.get("maxOutputChannels")

class AudioManager(object):
	def __init__(self , streaming = False):
		super().__init__()
		self._pa = pyaudio.PyAudio()
		try:
			self._inputDevice = AudioDevice(self._pa.get_default_input_device_info())
			# print("Default input device:", self._inputDevice)

		except IOError:
			print("Error: there is no default input device")
		self._streaming = streaming

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
		self._devices = []

		for i in range(self._pa.get_device_count()):
			# print("device:", self._pa.get_device_info_by_index(i))
			# print()
			device = self._pa.get_device_info_by_index(i)

			if(type == DeviceTypes.ALL_DEVICES):
				self._devices.append(AudioDevice(device))
			elif(type == DeviceTypes.INPUT_DEVICES and device.get("maxInputChannels") > 0):
				self._devices.append(AudioDevice(device))
			elif(type == DeviceTypes.OUTPUT_DEVICES and device.get("maxOutputChannels") > 0):
				self._devices.append(AudioDevice(device))

		return self._devices

	def getDevicesNameList(self, type=DeviceTypes.ALL_DEVICES):
		self.getDevices(type)
		names = []
		for device in self._devices:
			names.append(device.name())
		return names

	def getInputDevice(self) -> AudioDevice:
		return self._inputDevice

	def setInputDevice(self, device: AudioDevice):
		self._inputDevice = device

	def getDeviceFromIndex(self, index) -> AudioDevice:
		return AudioDevice(self._pa.get_device_info_by_index(index))

	def getDeviceFromName(self, deviceName):
		device_selected = AudioDevice(self._pa.get_default_input_device_info())

		self.getDevices()

		for device in self._devices:
			if(device.name() == deviceName):
				device_selected = device
				return device_selected

		return device_selected

	def getStreaming(self):
		return self._streaming

	def setStreaming(self, value):
		self._streaming = value

# JCR: Resources - https://www.youtube.com/watch?v=at2NppqIZok and https://github.com/aniawsz/rtmonoaudio2midi
class StreamProcessor(object):
	def __init__(self, input_device, \
				format = pyaudio.paFloat32, \
				input = True, \
				sample_rate = SAMPLE_RATE, \
				frames_per_buffer = FRAMES_PER_BUFFER, \
				channels = CHANNELS, \
				duration = DURATION):
		self._input_device = input_device
		self._format = format
		self._input = input
		self._sample_rate = sample_rate
		self._frames_per_buffer = frames_per_buffer
		self._channels = channels
		self._passes = 0
		self._duration = duration
		self._data = []

	def run(self):
		self._passes = 0
		self._data.clear()
		pya = pyaudio.PyAudio()
		self._stream = pya.open(
			format=self._format,
			channels=self._channels,
			input_device_index=self._input_device,
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
		next = pyaudio.paContinue

		# Calculate how many passes we require to do a full integration over DURATION
		# passes = selected_device.sampleRate() / FRAMES_PER_BUFFER * DURATION 
		# => passes * FRAMES_PER_BUFFER / selected_device.sampleRate() = DURATION
		calc = (frame_count/self._sample_rate) * self._passes
		# print("calculation:", calc)
		
		if(calc>= self._duration):
			next = pyaudio.paComplete
			# print("stop capturing:")

		self._data.append(data)
		self._passes += 1
		return (data, next)

	def getData(self):
		data = self._data
		return data

#Initialize AudioManager
am = AudioManager()

streaming = False
streamingMode = False

print("ALL SYSTEM INPUT AUDIO DEVICES:")
print()

devices = am.getDevices(DeviceTypes.INPUT_DEVICES)

for device in devices:
	sys.stdout.write(str(device.index()))
	sys.stdout.write(" ")
	sys.stdout.write(device.name())
	if(device.index() == am.getInputDevice().index()):
		sys.stdout.write(" (default)")
	sys.stdout.write('\n')

print()

# OSC callback functions
def getLufs(unused_addr, *args):
	# print("Running getLufs function!")
	# Initialize local variables
	loudness = -70.0
	if(type(args[0]) is float and args[0] >= 3.0):
		duration = args[0]
	else:
		duration = 3.0

	# print("Args:", args)

	audio_device = args[1]

	selected_device = am.getDeviceFromName(audio_device)

	total_frames_read = 0

	# Initialize meter
	state = R128State(
		selected_device.channels(DeviceType.INPUT_DEVICE),
		# 1,
		selected_device.sampleRate(),
		MeasurementMode.MODE_S
		)

	# Initialize Audio capture
	sp = StreamProcessor(
		selected_device.index(),
		channels=selected_device.channels(DeviceType.INPUT_DEVICE),
		sample_rate=selected_device.sampleRate(),
		duration=duration
		)

	# Capture audio frames for duration
	sp.run()
	data = sp.getData()

	# print("Data returned:", len(data))

	total_data = b''.join(data)
	data_samples = np.frombuffer(total_data,dtype=np.float32)

	block = data_samples.reshape(int(len(data_samples)/2), 2)
	# print("data samples:", int(len(data_samples)/2))

	frames_read = len(block)
	total_frames_read += frames_read

	for sample in block:
		# print("Sample:", len(sample), type(sample), sample)
		state.add_frames(sample, 1)

	if total_frames_read >= 3 * selected_device.sampleRate():
		# print("reached sample count:")
		inmediate_loudness = get_loudness_shortterm(state)
		# print("loudness:", inmediate_loudness)

		# Limiter lower output value
		if(inmediate_loudness < MIN_LOUDNESS):
			loudness = MIN_LOUDNESS
		else:
			loudness = inmediate_loudness

	#send the loundess as OSC
	client.send_message("/OSCLufs/lufs", loudness)

	del state

def streamLufs(unused_addr, *args):
	print("Streaming Lufs")

	loudness = -70.0
	# frames = []
	am.setStreaming(True)

	if(type(args[0]) is float and args[0] >= 0.1):
		duration = 0.2 # args[0]
	else:
		duration = 0.2

	audio_device = args[1]

	selected_device = am.getDeviceFromName(audio_device)

	total_frames_read = 0
	while am.getStreaming():
		# Initialize meter
		# meter = pyln.Meter(selected_device.sampleRate())
		state = R128State(
			selected_device.channels(DeviceType.INPUT_DEVICE),
			# 1,
			selected_device.sampleRate(),
			MeasurementMode.MODE_S
			)

		# print("State:", state)
		# print()

		# Initialize Audio capture
		sp = StreamProcessor(
			selected_device.index(),
			channels=selected_device.channels(DeviceType.INPUT_DEVICE),
			sample_rate=selected_device.sampleRate(),
			duration=duration
			)

		# Capture audio frames for duration
		sp.run()
		data = sp.getData()

		total_data = b''.join(data)
		data_samples = np.frombuffer(total_data,dtype=np.float32)

		# print("Sample size:", pyaudio.get_sample_size(pyaudio.paFloat32))
		# print()
		# print("Captured data:", len(data_samples), data_samples)
		# print()

		# block = [] #np.ndarray(shape=(2, int(len(data_samples)/2)), dtype=np.float32)
		# for i in range(int(len(data_samples)/2)):
		# 	# print("Sample pair:", i)
		# 	block.append(np.array([data_samples[i], data_samples[i+1]]))
		# 	i += 1

		block = data_samples.reshape(int(len(data_samples)/2), 2)

		# print("Block:", len(block), block)
		# print()

		frames_read = len(block)
		total_frames_read += frames_read

		# print("Block:", len(block), block)
		# print()
		for sample in block:
			# print("Sample:", len(sample), type(sample), sample)
			state.add_frames(sample, 1)

		# print("total data count:", len(total_data))
		# print("data samples count", len(data_samples))

		# Calculate loudness
		# inmediate_loudness = meter.integrated_loudness(data_samples) # measure loudness
		if total_frames_read >= 3 * selected_device.sampleRate():
			print("reached sample count:")
			inmediate_loudness = get_loudness_shortterm(state)
			print("loudness:", inmediate_loudness)

			# Limiter lower output value
			if(inmediate_loudness < MIN_LOUDNESS):
				loudness = MIN_LOUDNESS
			else:
				loudness = inmediate_loudness

			# #send the loundess as OSC
			# client.send_message("/OSCLufs/lufs", loudness)

		#send the loundess as OSC
		client.send_message("/OSCLufs/lufs", loudness)

	del state

def stopLufs(unused_addr):
	print("Stop sreaming Lufs")
	am.setStreaming(False)

def getAudioDevices(unused_addr, args):
	client.send_message("/OSCLufs/audio/devices", am.getDevicesNameList(_deviceTypeFromArgs(args)))

def getAudioDevicesCount(unused_addr, args):
	client.send_message("/OSCLufs/audio/devicesCount", am.getDevicesCount(_deviceTypeFromArgs(args)))

# OSC callback support functions
def _deviceTypeFromArgs(args):
	type_devices = DeviceTypes.ALL_DEVICES

	if(str(args).upper() == 'INPUT'):
		type_devices = DeviceTypes.INPUT_DEVICES
	elif(str(args).upper() == 'OUTPUT'):
		type_devices = DeviceTypes.OUTPUT_DEVICES
	
	return type_devices

# Main code
if __name__ == "__main__":
	# process arguments
	parser = argparse.ArgumentParser(description="OSCLufs an OSC Loudness meter")
	parser.add_argument(
		'--send_ip_address', '-ip', 
		dest='send_ip_address',
		default='127.0.0.1',
		help='Send ip address, default 127.0.0.1'
		)
	parser.add_argument(
		'--send_port', '-sp', 
		dest='send_port',
		default='1234',
		help='Send port, default: 1234'
		)
	parser.add_argument(
		'--receive_port', '-rp', 
		dest='receive_port',
		default='7070',
		help='Receive port, default: 7070'
		)

	args = parser.parse_args()
	send_ip = str(args.send_ip_address)
	send_port = int(args.send_port)
	in_port = int(args.receive_port)

	#Initialize audio manager
	am = AudioManager()
	
	#sending osc messages on
	client = udp_client.SimpleUDPClient(send_ip,send_port)
	sys.stdout.write("Opened Client on: ")
	sys.stdout.write(send_ip)
	sys.stdout.write(":")
	sys.stdout.write(str(send_port))
	sys.stdout.write('\n')

	#catches OSC messages
	dispatcher = dispatcher.Dispatcher()

	#lufs commands
	dispatcher.map("/OSCLufs/getLufs", getLufs)
	dispatcher.map("/OSCLufs/streamLufs", streamLufs)
	dispatcher.map("/OSCLufs/stopLufs", stopLufs)

	#audio commands
	dispatcher.map("/OSCLufs/audio/getDevices", getAudioDevices)
	dispatcher.map("/OSCLufs/audio/getDevicesCount", getAudioDevicesCount)
	
	#set up server to listen for osc messages
	server = osc_server.ThreadingOSCUDPServer((send_ip,in_port),dispatcher)
	print("Starting Server on {}".format(server.server_address))
	
	#Print API
	print("OSC Networking Established")
	print()
	print("OSC LUFS API")
	print()
	print("OSC API for Controlling OSCLufs:")
	print("/OSCLufs/getLufs {float integration time, string input device name}: Request a lufs reply")
	print()
	print("OSC API for Receiving Text from OSCLufs:")
	print("/OSCLufs/lufs {float num}: The lufs")
	print()
	print("OSC API to start streaming OSCLufs:")
	print("/OSCLufs/streamLufs {float refresh rate, string input device name, string measurement mode}: Start streaming lufs reply")
	print()
	print("OSC API to stop streaming OSCLufs:")
	print("/OSCLufs/stopLufs: Stop streaming lufs")
	print()
	print("OSC API for Receiving Text from OSCLufs:")
	print("/OSCLufs/lufs {float num}: The lufs")
	print()
	# Todo: To be implemented 
	# print("OSC API for Receiving Text from OSCLufs:")
	# print("/OSCLufs/maxLufs {float num}: The maximum value of lufs")
	# print()

	#Print Audio API
	print("OSC Audio API")
	print()
	print("OSC API for Getting audio input devices count:")
	print("/OSCLufs/audio/getDevicesCount {string ALL|INPUT|OUTPUT}: Request input devices count")
	print()
	print("OSC API for Receiving Text from OSCLufs:")
	print("/OSCLufs/audio/devicesCount {integer num}: Input devices count")
	print()
	print("OSC API for Getting audio input devices:")
	print("/OSCLufs/audio/getDevices {string ALL|INPUT|OUTPUT}: Request input devices")
	print()
	print("OSC API for Receiving Text from OSCLufs:")
	print("/OSCLufs/audio/devices {List string}: Input devices name list")
	print()
	#begin the infinite loop
	server.serve_forever()
