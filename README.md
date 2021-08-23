# OSCLufs
OSCLufs is an audio analytics tool that posts LUFS data to the network via the Open Sound Control (OSC) protocol. 

## Source Dependencies
You'll need to install these external libraries if you plan to build from source:

PythonOSC for using Open Sound Control: https://pypi.org/project/python-osc/

PyAudio for getting data from the audio bus: https://pypi.org/project/PyAudio/

Python Sound File for decoding the audio buffer: https://pypi.org/project/SoundFile/

Pyebur128 for calculating the short term lufs: https://github.com/jodhus/pyebur128

## Build
Releases are built with pyinstaller using: pyinstaller --hidden-import scipy.spatial.transform._rotation_groups --hidden-import scipy.special.cython_special OSCLufs.py

Anaconda is used due to issues with PyAudio, pip installed via Anaconda to manage incorporation of pyloudnorm

## OSC Commands

OSC API for Controlling OSCLufs:

/OSCLufs/getLufs {float integration time, string input device name}: request for the lufs data reply
Parameters:
    1.- Integration time for loudness calculation, range:
        * Min: The minimum integration time is 3 seconds.
        * Max: No maximum value defined
    2.- Input device name, if name no correspond to one of retrievef list in /OSCLufs/audio/devices
        the system selected input device is used

By default, you send these commands to port 7070. Then, you should listen for the following OSC message:
/OSCLufs/lufs {float data}: The current lufs reading from the buffer

/OSCLufs/streamLufs {float refresh rate, string input device name}: start stream of lufs data
Parameters:
    1.- Integration time for loudness calculation, range:
        * Min: The minimum integration time is 3 seconds.
        * Max: No maximum value defined
    2.- Input device name, if name no correspond to one of retrievef list in /OSCLufs/audio/devices
        the system selected input device is used

/OSCLufs/stopLufs: stop the stream of lufs data

OSC Audio API for Controlling OSCLufs audio input:

/OSCLufs/audio/getDevicesCount {string ALL|INPUT|OUTPUT}: Requests for the audio inputs count
Parameters:
    1.- Type of devices of type string, available values:
        * ALL: Search against all devices connected.
        * INPUT: Only input devices connected.
        * OUPUT: Only output devices connected.

Listen for the following OSC message:
/OSCLufs/audio/devicesCount {integer num}

/OSCLufs/audio/getDevices {string ALL|INPUT|OUTPUT}: Request input devices name
Parameters:
    1.- Type of devices of type string, available values:
        * ALL: search against all devices connected.
        * INPUT: only input devices connected.
        * OUPUT: only output devices connected.

Listen for the following OSC message:
/OSCLufs/audio/devices {List string}: Input devices name list