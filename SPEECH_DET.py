import webrtcvad
import pyaudio
import sys
import time
import wave
from uuid import uuid4

print("Voice Activity Monitoring")
print("1 - Activity Detected")
print("_ - No Activity Detected")
print("X - No Activity Detected for Last IDLE_TIME Seconds")
input("Press Enter to continue...")
print("\nMonitor Voice Activity Below:")

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
FRAMES_PER_BUFFER = 320

vad = webrtcvad.Vad(3)

pa = pyaudio.PyAudio()
stream = pa.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=FRAMES_PER_BUFFER)

inactive_session = False
inactive_since = time.time()
frames = []
while True:
    data = stream.read(FRAMES_PER_BUFFER)

    is_active = vad.is_speech(data, sample_rate=RATE)
    
    idle_time = 2
    if is_active:
        inactive_session = False
    else:
        if inactive_session == False:
            inactive_session = True
            inactive_since = time.time()
        else:
            inactive_session = True

    if (inactive_session == True) and (time.time() - inactive_since) > idle_time:
        sys.stdout.write('X')
        
        frames.append(data)

        audio_recorded_filename = f'RECORDED-{str(time.time())}-{str(uuid4()).replace("-","")}.wav'
        wf = wave.open(audio_recorded_filename, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(pa.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()

        time.sleep(5)
        inactive_session = False
    else:
        sys.stdout.write('1' if is_active else '_')
    
    frames.append(data)

    sys.stdout.flush()

stream.stop_stream()
