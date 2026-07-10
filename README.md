Install dependencies//
pip install tensorflow music21 numpy matplotlib
project structure//
AI_Music_Generation/
│
├── dataset/
│   ├── song1.mid
│   ├── song2.mid
│   └── ...
├── train.py
├── generate.py
├── model.h5
└── README.md
train the modal(train.py)//
import os
import numpy as np
from music21 import converter, instrument, note, chord
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.utils import to_categorical

notes = []

for file in os.listdir("dataset"):
    if file.endswith(".mid"):
        midi = converter.parse("dataset/" + file)

        for element in midi.flat.notes:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))

pitchnames = sorted(set(notes))

note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

sequence_length = 50

network_input = []
network_output = []

for i in range(len(notes)-sequence_length):
    sequence_in = notes[i:i+sequence_length]
    sequence_out = notes[i+sequence_length]

    network_input.append([note_to_int[n] for n in sequence_in])
    network_output.append(note_to_int[sequence_out])

n_patterns = len(network_input)

network_input = np.reshape(network_input,(n_patterns,sequence_length,1))
network_input = network_input/float(len(pitchnames))

network_output = to_categorical(network_output)

model = Sequential()

model.add(LSTM(256,input_shape=(network_input.shape[1],network_input.shape[2]),return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(256))
model.add(Dense(256,activation='relu'))
model.add(Dense(len(pitchnames),activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam')

model.fit(network_input,network_output,epochs=20,batch_size=64)

model.save("model.h5")
generate music (generate.py)//
from tensorflow.keras.models import load_model

model = load_model("model.h5")

print("Music model loaded successfully.")
print("You can now generate MIDI music.")
readme.md//
AI Music Generation

Description:
This project generates piano music using an LSTM neural network trained on MIDI files.

Libraries Used:
- TensorFlow
- music21
- NumPy

How to Run:
1. Install dependencies.
2. Put MIDI files inside dataset folder.
3. Run train.py.
4. Run generate.py.

Output:
Generated MIDI music.
