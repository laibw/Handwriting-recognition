# <ins>Documentation of esp32 handwriting recognition system</ins>

Here are details of the step by step guide to creating a handwriting recognition system utilizing ESP32 microcontroller, machine learning with Tensorflow, touch screen input, and Arduino programming. Items required are listed below:

-   ESP32 dev kit (with GPIO pins)
-   ILI9488 TFT Module
-   Breadboard
-   Jumper wires
-   Micro USB Type-B to USB Type-A cable
-   Computer with USB Type-A socket
-   Red and Green LEDs (optional)
-   Resistors (optional)

## <ins>Handwriting system</ins>

First, the handwriting input system uses ILI9488 3.5’’ SPI TFT Module. This is a 320x480 TFT touch-enabled screen that will be used as a writing pad. An ESP32 Dev Module will be used as the microcontroller. Breadboard and jumper wires will be used as connections between ESP32 and screen.

Install the Arduino IDE to be able to program the ESP32. The version used in this project is Arduino 1.8.19 for reliability. Newer versions of the IDE should also be usable. To begin, open a new sketch in the IDE and go to File-\>Preferences. Under Additional Boards Manager URLs paste in the following:

<https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json>

<http://arduino.esp8266.com/stable/package_esp8266com_index.json>

This will provide the IDE with the board manager settings for various ESP32 configurations, which will enable the IDE to properly communicate with the microcontroller. Go to Tool-\>Board: and choose the respective ESP32 board that will be used for this project. This project will use ESP32 Dev Module.

This is the specific configuration that will be used for this project:

![image](https://github.com/user-attachments/assets/6e307fbd-31fe-4e98-a06c-0526a9dca52e)

The Erase All Flash Before Sketch Upload option should be enabled when wanting to have a clean slate and removes all previous data stored within the SPIFFS internal storage (for example recalibrating screen). The partition scheme: “Huge APP (3MB No OTA/1MB SPIFFS)” used will be suitable storing large programs just in case the program size gets too big.

The main library that will be used for enabling the TFT screen is TFT_eSPI, which can be installed via the library manager. After installing the TFT_eSPI library, go to the library folder installation location and access User_Setup.h and User_Setup_Select.h under Arduino/libraries/TFT_eSPI, and Setup21_ILI9488.h under /User_Setups.

In User_Setup.h, comment out the line (line45):
```c++
//#define ILI9341_DRIVER       // Generic driver for common displays`
```
<br />

Uncomment this line (line54):
```c++
#define ILI9488_DRIVER     // WARNING: Do not connect ILI9488 display SDO to MISO if other devices share the SPI bus (TFT SDO does NOT tristate when CS is high)
```
<br />

Comment out these lines (line170-176):
```c++
//#define TFT_MISO  PIN_D6  // Automatically assigned with ESP8266 if not defined
//#define TFT_MOSI  PIN_D7  // Automatically assigned with ESP8266 if not defined
//#define TFT_SCLK  PIN_D5  // Automatically assigned with ESP8266 if not defined

//#define TFT_CS    PIN_D8  // Chip select control pin D8
//#define TFT_DC    PIN_D3  // Data Command control pin
//#define TFT_RST   PIN_D4  // Reset pin (could connect to NodeMCU RST, see next line)
```
<br />

Uncomment this line (line230):
```c++
#define TOUCH_CS 21     // Chip select pin (T_CS) of touch screen
```
<br />

Go to User_Setup_Select.h and uncomment this line (line52):
```c++
#include <User_Setups/Setup21_ILI9488.h>           // Setup file for ESP32 and ILI9488 SPI bus TFT
```
<br />

Go to Setup21_ILI9488.h and make sure the values are as shown below:
```c++
#define USER_SETUP_ID 21

#define ILI9488_DRIVER

//#define TFT_INVERSION_OFF

#define TFT_MISO 19 // (leave TFT SDO disconnected if other SPI devices share MISO)
#define TFT_MOSI 23
#define TFT_SCLK 18
#define TFT_CS    15  // Chip select control pin
#define TFT_DC    2  // Data Command control pin
#define TFT_RST   4  // Reset pin (could connect to RST pin)
```
<br />

Save all the changes made and return back to the Arduino IDE.

Now that the user setup for the library has been properly configured for our hardware, we can begin coding for the touchscreen functionality. Go to examples and under TFT_eSPI-\>480 x 320 open Touch_Controller_Demo. This example sketch will be the starting point and template for touchscreen controls. The Touch_Controller_Demo consists of 2 parts, calibrating the screen and drawing with touch input. The code in the void setup() portion checks for calibration values within the ESP32 SPIFFS internal storage and uses it to calibrate the screen touch sensors. If there is no calibration value available, the program starts a calibration run on the screen that prompts the user to touch the 4 corners of the screen to calibrate. The void loop() section then takes user touch input on the screen and calculates the coordinates touched. The program draws a pixel at the coordinate and this runs continuously.

Before testing this sketch, we will have to setup the hardware. Place the ESP32 pins between 2 breadboards and the ILI9488 screen pins separate from the ESP32. Using jumper cables, connect the screen VCC pin to ESP32 3.3V output. Connect the GND pin on the screen to the GND pin on the ESP32. The CS pin is connected to GPIO 15 pin, RESET pin is connected to GPIO 4 pin, DC/RS pin is connected to GPIO 2 pin, SDI(MOSI) pin is connected to GPIO 23 pin, SCK pin is connected to GPIO 18 pin, LED pin is connected to 3.3V pin, T_CLK pin is connected to GPIO 18 pin, T_CS pin is connected to GPIO 21, T_DIN pin is connected to GPIO 23 pin, T_DO pin is connected to GPIO 19 pin. Do not connect SDO(MOSI) pin and T_IRQ pin to anything. The GPIO pins numbers can be changed as long as it is also changed within Setup21_ILI9488.h and User_Setup.h to the corresponding values. After all the connections have been completed, plug in the micro USB type-B cable into the ESP32 and connect it to a USB port on the computer. Make sure the port section under tools is available and selected (example COM 5). Enable Erase All Flash Before Sketch Upload option to ensure there is no data from previous uses interfering. Verify and upload the example sketch to the ESP32 and check whether it is working as expected. The screen should light up and prompt the user to perform calibration as it is the first time running and have no calibration data. Tap the 4 red boxes at the corners to complete calibration and then begin drawing on the screen with a pointy object such as a plastic pen or finger. It should leave behind multicoloured pixels wherever it is touched. This confirms the display and touch functionality is working. Notice the x and y coordinates displayed on the screen, it represents where on the screen is touched and ranges between 0 to 320 on the y-axis and 0 to 480 on the x-axis. If the coordinate values do not match the area touched, then it might mean the calibration is off and needs to be recalibrated by reuploading the sketch with Enable Erase All Flash Before Sketch Upload option enabled. Once the calibration is accurate then this option can be disabled for future uploads to avoid recalibration.

![image](https://github.com/user-attachments/assets/e6ef7113-05de-4bc3-8774-90f0eb2c8ee5)

One thing to notice is the trails left behind is rather thin and faint, which is not useful for handwriting recognition. In order for the handwriting to be usable, it must be converted into a 28x28 bitmap. Rather than working with a 28x28 pixel area, the drawing region will be 280x280 pixels and subdivided into 10x10 pixels chunks as the pixels for handwriting bitmap. The code for the current sketch will have to be modified for the new requirements. Go to the link below and open touch_test/touch_test.ino:

[Handwriting-recognition/touch_test at main · laibw/Handwriting-recognition (github.com)](https://github.com/laibw/Handwriting-recognition/tree/main/touch_test)

The difference between this code and the previous code is that a 28x28 2D array is initialised at the beginning of the sketch which will be used to store our bitmap. The array is initialised to all 0s to represent an empty image. At the end of void setup() a red rectangular frame is drawn to represent the drawing area.

```c++
    if(x<280&&y<280){
       x=x/10;
       y=y/10;
       d[x][y]=1;
       x=x*10;
       y=y*10;
       tft.fillRect(x, y, 10, 10, TFT_BLACK);
    }else{
       for(int i=0;i<height;i++){
          Serial.println();
          for(int j=0;j<width;j++){
            Serial.printf("%d ",d[j][i]);
          }
       }   
    }
```

This bit of code in void loop() purpose is to calculate which 10x10 subdivision is being touched and filling in the particular spot in the array. Example: if position is (250,200), then the array element [25][20] is set to 1. The algorithm takes advantage of the rounding done when dividing integers such that the remainder/decimal is ignored and the number is rounded down, so position 0-9 is counted as position 0, 10-19 is position 1 etc. After filling in the array, the area on the screen is also filled in with a 10x10 black box as to represent the drawn pixel, with the position of the box determined by the coordinates touched. If the region touched is not within the drawing region then the array(bitmap) info is sent to the Serial Monitor where we can confirm its validity.

![image](https://github.com/user-attachments/assets/2e990573-3de3-4b6c-98a3-05a80b6f7c4a)|![image](https://github.com/user-attachments/assets/f180f195-c69b-4113-a5ec-7c69bcb7ae19)

The user should be able to produce images similar to the above by drawing within the red box region, and after clicking outside the box would produce the image on the right in the Serial Monitor. Storing the image as an array will be the main method of representing the handwriting, which can then be used as input on neural networks for handwriting recognition.

## <ins>Training neural network model for handwriting recognition</ins>

The target of our handwriting recognition system is to be able to recognise handwritten alphabets. This will require a neural network that has been trained on handwritten alphabets to detect the users handwriting and output the closest matched alphabet. In order to train the model, high quality training data with labels is required. The link below contains a downloadable csv file that contains 300000+ examples of handwritten alphabets in grayscale format with labels.

[A-Z Handwritten Alphabets in .csv format (kaggle.com)](https://www.kaggle.com/datasets/sachinpatel21/az-handwritten-alphabets-in-csv-format)

Save the download into google drive as it will be useful later.

Open Google Colab (or Jupyter notebook on local machine, recommended using colab for computation resources) and open Modeltraining.ipynb file below:

[Handwriting-recognition/Modeltraining.ipynb at main · laibw/Handwriting-recognition (github.com)](https://github.com/laibw/Handwriting-recognition/blob/main/Modeltraining.ipynb)

This is a python notebook which contains code to train the handwriting recognition model with Tensorflow/Tensorflowlite. Tensorflow is a machine learning framework that helps with simplifying machine learning processes, while Tensorflowlite is a lightweight version of Tensorflow suitable for edge and embedded devices like ESP32. The notebook is spilt into code blocks that will be run sequentially and separately for easier understanding.

Starting with the first code block:
```python
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import CategoricalCrossentropy
from sys import getsizeof
from google.colab import files
from google.colab import drive
```
This imports all the library packages that will be used and includes useful packages like Tensorflow, Numpy etc.  
<br />

```python
output_labels = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
```
This line is used to map output numbers to an alphabet which will be useful later.  
<br />

```python
def get_file_size(file_path):
  return os.path.getsize(file_path)
```
This defines a function to calculate file size.  
<br />

```python
drive.mount('/content/drive')
```
This line is for mounting google drive onto google colab environment so that files in google drive can be transferred over to colab. (Manually transferring files from local machine to google colab is very slow, google drive is much faster).
![image](https://github.com/user-attachments/assets/52ab56eb-fc41-442e-ac06-01505cb81b05)
After running this line, a prompt will show up asking for permission to access google drive. Follow through all the steps and connect google colab to the google drive account which contains the csv data.  
<br />

```python
data_path = '/content/drive/MyDrive/handwritten_data_785.csv'
df = pd.read_csv(data_path)
df.shape
```
Here the data from the csv file in google drive is read and stored in variable df (Note that the csv file name here is handwritten_data_785.csv, but it could be different and has to be changed in the code accordingly). df.shape shows the shape of the df: (372037, 785). 372037 examples of alphabet handwriting and 784 pixel value +1 label value.  
<br />

```python
features = df.values[:,1:]
labels = df.values[:,0]
```
This splits the data into pixel values(features) and labels using array slicing. [:,1:] means all rows and second to last column, [:,0] means all rows and first column (since the label value is stored in the first column).  
<br />

```python
features = features / 255.
features=np.ceil(features)
labels=keras.utils.to_categorical(labels)
```
This sections deals with preprocessing the data to be more suitable for training. First the features are divided by 255 to convert the range from 0-255 to 0-1 which will be easier to train. The pixel values are then converted from grayscale to binary using np.ceil() function which makes all non 0 pixels 1. This is done to match the input method of the handwriting system as a binary bitmap, which will produce a model that is better suited for the current usage. The labels currently are in the form of 0-25 to represent alphabets, but it will be more useful if it’s a one-hot encoded array with all elements as 0 except the position corresponding to the label itself as 1. This is done using the keras.utils.to_categorical() function. A label that was originally 13 for example will become a 1D array of all 0 except index 13 which is 1.  
<br />

```python
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.25, random_state=42)
```
This line splits the labels and features into training and testing datasets randomly in a 75:25 split. 42 is the random seed used so that the dataset can be replicated exactly between examples. x are features and y are labels. Splitting the dataset is important to avoid overfitting the training data set during training.  
<br />

```python
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
```
Prints out the shape of the dataset. x_train should have 279027 rows and 784 columns(28x28) while x_test have 93010 rows and the same amount of columns. y_train and y_test should have the same amount of rows are the x counterparts and 26 columns representing the one-hot array.  
<br />

```python
model=keras.Sequential([
    Dense(128,activation='relu',input_shape=(784,)),
    Dense(26)
])
```
This block of code defines the model layers that will be implemented. Dense means a fully interconnected layer of nodes with the first parameter being the number of nodes. The activation function used is relu(rectified linear) which is a commonly used activation function for machine learning. The input shape is the number of features that is used as input, in this case 784 pixels. The model here is a simple sequential neural network with 2 layers having 128 and 26 nodes respectively. The second layer has the same amount of nodes as the output(26 alphabets). The output node with the highest activation will be considered as the predicted alphabet.  
<br />

```python
model.summary()
```
The number of parameters can be determined by the number of interconnections between nodes as these connections holds the model weights(parameters). Total parameters=(784+1(bias))x128+(128+1(bias))x26=103834 which matches the summary.  
<br />

```python
model.compile(optimizer='adam',
              loss=CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
```
After defining the model structure, the model is set up for training using model.compile(). The optimizer is set to adam, and the loss function by which the model is trained to minimise is CategoricalCrossentropy with from_logits set to true because the output values have not undergone softmax and are considered logits. A metric to pay attention to is accuracy which is the rate of a correct prediction.  
<br />

```python
del df
del features
del labels
```
This block is run to delete redundant data to save memory (colab offers only 12GB on free version). If memory is not enough, the runtime will crash, and training cannot complete.  
<br />

```python
model.fit(x_train,y_train,epochs=10)
```
This line will begin running the training algorithm for 10 epochs(cycles) by tuning the weights based on minimizing the loss function. The loss value will decrease as training goes on and accuracy will increase.  
<br />

```python
test_loss,test_accuracy = model.evaluate(x_test,y_test, verbose=2)
print("Accuracy:",format(round(100*test_accuracy,2)))
```
This is to evaluate the model against a separate testing dataset that the model has never seen before. If the accuracy does not differ significantly from training accuracy then the model is considered to have not overfitted. In this case the model achieve over 96% accuracy on the testing dataset.  
<br />

```python
model_name='tf_model_AZ_handwriting.keras'
model.save(model_name)
```
Now save the model as a backup. The model can be found under the content folder in google colab.  
<br />

```python
print(get_file_size(model_name))
```
This shows the size of the model, in this case 1.2MB which is too big for ESP32. The model has to be shrunk in order to deploy on the microcontroller’s limited memory.  
<br />

```python
test_image=x_test[0:10]
test_label=y_test[0:10]
classes_y=np.argmax(test_label,axis=1)
predictions=model.predict(test_image)
classes_x=np.argmax(predictions,axis=1)
print('Actual classes: ',end="")
for q in classes_y:
  print(output_labels[q],end=" ")
print('\nPredicted classes: ',end="")
for p in classes_x:
    print(output_labels[p],end=" ")
```
A piece of code to demonstrate how the model works. By taking a few sample features from the testing dataset and running through the model, the model will output different an array of 26 values representing how close the handwriting is to each alphabet. The highest value position is then translated into the corresponding alphabet as the chosen predicted alphabet.  
<br />

![image](https://github.com/user-attachments/assets/325234af-69b7-4fff-a17b-69fda733f76d)

As shown the model can correctly predict the samples tested.  

In order to run the model on ESP32, the model has to be smaller to fit into the limited memory. As good way to do this is to convert the Tensorflow model into a Tensorflowlite model. Tensorflowlite does a few optimizations and quantization to reduce the space required to store the parameter(weights) which the model is mainly comprised of.  
<br />

```python
tflite_model_name="tf_lite_model_AZ_handwriting.tflite"
tf_lite_converter=tf.lite.TFLiteConverter.from_keras_model(model)
tf_lite_converter.optimizations = [tf.lite.Optimize.DEFAULT]
def representative_dataset_generator():
  for value in x_test:
  # Each scalar value must be inside of a 2D array that is wrapped in a list
    yield [np.array(value, dtype=np.float32, ndmin=2)]
tf_lite_converter.representative_dataset = representative_dataset_generator
tflite_model=tf_lite_converter.convert()
```
This block of code is to convert the Tensorflow model into a tensorflowlite model. A converter is set up that takes in the tensorflow model and applies various optimizations to reduce the size of the model. The main optimization used is quantization which converts the weights(which are usually stored as floats) into a smaller data type without losing much accuracy. In order to apply quantization a representative dataset should be provided for the algorithm to estimate and calibrate the range weights so that it can optimize the size. The representative dataset is taken from the test dataset in this case.  
<br />

```python
open(tflite_model_name,"wb").write(tflite_model)
```
This line saves the tensorflowlite model as a tflite file which can be found under the content folder of google colab. The output shows the size of the converted model which is around 100kB which is an order of magnitude smaller and more suitable to be run on the ESP32.  
<br />

```python
interpreter=tf.lite.Interpreter(model_path=tflite_model_name)
```
```python
input_details=interpreter.get_input_details()
output_details=interpreter.get_output_details()
print(input_details[0]['shape'])
print(input_details[0]['dtype'])
print(output_details[0]['shape'])
print(output_details[0]['dtype'])
```
To test the tensorflowlite version of the model, it has to be run through an interpreter. The input tensor(array) accepts only 1 sample at a time as can be seen in the input_details shape. To test a large scale dataset all at once the input tensor has to be reshaped to accept more samples at once.  
<br />

```python
interpreter.resize_tensor_input(input_details[0]['index'],(93010,784))
interpreter.resize_tensor_input(output_details[0]['index'],(93010,26))
interpreter.allocate_tensors()
input_details=interpreter.get_input_details()
output_details=interpreter.get_output_details()
print(input_details[0]['shape'])
print(input_details[0]['dtype'])
print(output_details[0]['shape'])
print(output_details[0]['dtype'])
```
This block is responsible for reshaping the input tensor to take in multiple samples at once. The input tensor now has 93010 rows, same as the number of test samples.  
<br />

```python
test_image_tflite=np.array(x_test,dtype=np.float32)
```
Prepare a testing dataset that has a compatible datatype with the input tensor(np.float32).  
<br />

```python
interpreter.set_tensor(input_details[0]['index'],test_image_tflite)
interpreter.invoke()
tflite_prediction=interpreter.get_tensor(output_details[0]['index'])
prediction_class=np.argmax(tflite_prediction,axis=1)
```
```python
actual_class=np.argmax(y_test,axis=1)
acc=accuracy_score(prediction_class,actual_class)
print(acc)
```
This code invokes the interpreter to perform inference on the input data and then takes the predicted values from the output tensor to compare against the actual values. As can be seen, the accuracy is about 96% which is similar to the tensorflow model and has minimal loss of accuracy from conversion. This is desirable and proves that tensorflowlite models are viable to be used.  
<br />

```
!apt-get -qq install xxd
```
```
!xxd -i /content/tf_lite_model_AZ_handwriting.tflite > tf_lite_model_AZ_handwriting.cc
```
Finally, convert the tflite model into a C++ file by using xxd command to dump the hex data into a c array. This is done to more easily include the model in the Arduino sketch. Download the cc file on the local machine.  

After this the training phase is done and integration of handwriting system and handwriting recognition model can be performed.

## <ins>Complete Handwriting Recognition System</ins>
In order to use the tensorflowlite model in Arduino sketch, it has to be included as a header file. Create a header file: tf_lite_model_AZ_handwriting.h
Inside the header file add the following lines of code:
```c++
extern const unsigned char g_model[];
extern const int g_model_len;
```
This will tell your program that includes this header file that there are variables with these names already declared in another cpp file.  
<br />
Open the cc file and make sure the variable names is the same as in the header file.  
Self include the header file in the cc file(to prevent definition mismatch) and add alignas(8) infront of the array declaration:
```c++
#include "tf_lite_model_AZ_handwriting.h"
alignas(8) const unsigned char g_model[] = {
```
Save the cc file as cpp file because Arduino IDE only recognises cpp file (despite being the same thing).  
Now the source and header files for the model is properly formatted, it can be included into a new sketch folder, using a copy of touch_test.ino as the basis for the next phase.  
<br />
The source code for Complete Handwriting Recognition System is in the link below:  
[Handwriting-recognition/touch_AZ_example_Copy at main · laibw/Handwriting-recognition (github.com)](https://github.com/laibw/Handwriting-recognition/tree/main/touch_AZ_example_Copy)  
<br />
Building upon the code for touch_test, additional libraries are added to include tensorflowlite support.
```c++
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tf_lite_model_AZ_handwriting.h"
#include "FS.h"
#include <SPI.h>
#include <TFT_eSPI.h>
```
Various libraries related to tensorflowlite is included for different purposes, such as operation resolvers, interpreter, setup and logging. The model header file is also included (tf_lite_model_AZ_handwriting.h) along with the libraries for the screen.  
<br />
```c++
TFT_eSPI tft = TFT_eSPI();
#define CALIBRATION_FILE "/calibrationData"

// Globals, used for compatibility with Arduino-style sketches.
namespace {
const tflite::Model *model = nullptr;
tflite::MicroInterpreter *interpreter = nullptr;
TfLiteTensor *input = nullptr;
TfLiteTensor *output = nullptr;
constexpr int kTensorArenaSize = 10000;
uint8_t tensor_arena[kTensorArenaSize];
const int width = 28;
const int height = 28;
float d[width][height];//image array
int draw=1;
const int redled=25;
const int greenled=27;
}  // namespace
```
Here all the variables and objects are declared. Note the redled and greenled variable values as this are the GPIO pins that is connected to the red and green LEDs. kTensorArenaSize is the size of the memory allocated to the tensors required to run the model. This can be smaller than the model size due to efficient allocation and deallocation management during runtime. The value can be reduced by trial and error until the program cannot run, after which it is set to the smallest usable value.  
<br />
The beginning of void setup() is similar to the code in touch_test where the image array is initialised to all 0s and the screen the calibrated.
```c++
  //draw boxes and words for GUI
  tft.fillScreen((0xFFFF));
  tft.drawRect(0,0,281,281,TFT_RED);
  tft.fillRect(300,100,50,50,TFT_GREEN);
  tft.setCursor(351, 100, 2);
  tft.print("Recognise alphabet");
  tft.fillRect(300,200,50,50,TFT_RED);
  tft.setCursor(351, 200, 2);
  tft.print("Reset");
  tft.fillRect(300,260,40,40,TFT_GREEN);
  tft.setCursor(300, 300, 2);
  tft.print("Draw");
  tft.fillRect(340,260,40,40,TFT_RED);
  tft.setCursor(340, 300, 2);
  tft.print("Erase");
```
After calibration, draw a few rectangles on the screen that represent different interactable regions such as drawing region, reset button, recognise button, draw and erase button.  
<br />
```c++
  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf(
      "Model provided is schema version %d not equal to supported "
      "version %d.",
      model->version(), TFLITE_SCHEMA_VERSION
    );
    return;
  }
```
After completing the screen setup, begin the tensorflowlite setup. The first line creates a model object from using the model data in the model source file (tf_lite_model_AZ_handwriting.cpp). It checks whether the model version is compatible with the current schema of the library, and prints an error message if not.  
<br />
```c++
  // Pull in only the operation implementations we need.
  static tflite::MicroMutableOpResolver<3> resolver;
  if (resolver.AddFullyConnected() != kTfLiteOk) {
    MicroPrintf("Failed resolver.AddFullyConnected()");
    return;
  }
  if (resolver.AddQuantize() != kTfLiteOk) {
    MicroPrintf("Failed resolver.AddQuantize()");
    return;
  }
  if (resolver.AddDequantize() != kTfLiteOk) {
    MicroPrintf("Failed resolver.AddDequantize()");
    return;
  }
```
This section adds the required operations to the MicroMutableOpResolver which  has the  instructions to perform the calculations required to run the model. This model is a fully connected perceptron type neural network and requires the FullyConnected operation. Since the model has been quantized during conversion to tensorflowlite, quantize and dequantize operations are also required. Error message will be displayed if this step fails.  
<br />
```c++
  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;
```
Similarly to Python, an interpreter is required to run the tensorflowlite model. The interpreter requires information such as the model itself, the operations inside the opresolver, the tensor arena and size of the arena.  
<br />
```c++
  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    MicroPrintf("AllocateTensors() failed");
    return;
  }
```
Allocate the memory for the tensor arena.  
<br />
```c++
  // Obtain pointers to the model's input and output tensors.
  input = interpreter->input(0);
  output = interpreter->output(0);
  pinMode(redled,OUTPUT);
  pinMode(greenled,OUTPUT);
```
Obtain pointers to the input and output tensors for easy access. Also initialise the GPIO pins for the LEDs.  
<br />
Moving on to void loop()
```c++
  if(draw==1){
    digitalWrite(greenled,HIGH);
    digitalWrite(redled,LOW);
  }else{
    digitalWrite(greenled,LOW);
    digitalWrite(redled,HIGH);    
  }
```
The first part determines whether the green or red LED will light up based on the current mode of drawing. If under drawing mode, green lights up; if under erase mode, red lights up. Remember to connect the LEDs in series with resistor to the respective GPIO pins.  
<br />
```c++
  uint16_t x, y;//touch coordinates
  if (tft.getTouch(&x, &y)) {
    //tft.fillRect(300,60,100,100,TFT_WHITE);
    tft.setCursor(281, 5, 2);
    tft.printf("x: %i     ", x);
    tft.setCursor(281, 20, 2);
    tft.printf("y: %i    ", y);//display coordinates
```
Detects whether the screen is touched and display the coordinates.  
<br />
```c++
    //draw button
    if(300<x && x<340 && 260<y && y<300){
      draw=1;

    //erase button
    }else if(340<x && x<380 && 260<y && y<300){
      draw=0;
```
Checks if the draw or erase button is touched and sets the mode using draw variable.  
<br />
```c+=
    //drawing
    }else if(x<280&&y<280&&draw==1){
      x=x/10;
      y=y/10;
      if(x<27&&y<27){
        d[x][y]=1.f;
        d[x+1][y]=1.f;
        d[x][y+1]=1.f;
        d[x+1][y+1]=1.f;
        x=x*10;
        y=y*10;
        tft.fillRect(x, y, 20, 20, TFT_BLACK);
      }else if(x==27&&y<27){
        d[x][y]=1.f;
        d[x][y+1]=1.f;
        x=x*10;
        y=y*10;
        tft.fillRect(x, y, 10, 20, TFT_BLACK);
      }else if(x<27&&y==27){
        d[x][y]=1.f;
        d[x+1][y]=1.f;
        x=x*10;
        y=y*10;
        tft.fillRect(x, y, 20, 10, TFT_BLACK);
      }else{
        d[x][y]=1.f;
        x=x*10;
        y=y*10;
        tft.fillRect(x, y, 10, 10, TFT_BLACK);
      }
```
When drawing in the drawing region, the cursor will draw 2x2 pixels(subregions) to better emulate actual handwriting which has thickness. This is also reflected in array and screen display. The if-else statements check for boundary conditions to avoid drawing out of bounds of the drawing region/array. This block of code is only executed if the mode is drawing mode(draw==1) and within the drawing region.  
<br />
```c++
    //erasing
    }else if(x<280&&y<280&&draw==0){
      x=x/10;
      y=y/10;
      if(x<27&&y<27){
        d[x][y]=0;
        d[x+1][y]=0;
        d[x][y+1]=0;
        d[x+1][y+1]=0;
        x=x*10;
        y=y*10;
        tft.fillRect(x, y, 20, 20, TFT_WHITE);
      }else if(x==27&&y<27){
        d[x][y]=0;
        d[x][y+1]=0;
        x=x*10;
        y=y*10;
        tft.fillRect(x, y, 10, 20, TFT_WHITE);
      }else if(x<27&&y==27){
        d[x][y]=0;
        d[x+1][y]=0;
        x=x*10;
        y=y*10;
        tft.fillRect(x, y, 20, 10, TFT_WHITE);
      }else{
        d[x][y]=0;
        x=x*10;
        y=y*10;
        tft.fillRect(x, y, 10, 10, TFT_WHITE);
      }
```
This section is similar to above except for erasing in a 2x2 region, useful for erasing unwanted parts of the drawing. This only executes when in erase mode(draw==0) and within the drawing region.  
<br />
```c++
    //click green button
    }else if(x>300&&x<350&&y>100&&y<150){
      //send image array to serial monitor
      for(int i=0;i<height;i++){
        Serial.println();
        for(int j=0;j<width;j++){
          if(d[j][i]==0){
            Serial.print("  ");
          }else{
            Serial.print("@@");
          }
        }
      }
```
This part runs when the green button(region) is touched which starts the handwriting recognition process. First the image is displayed in the Serial Monitor as an ASCII bitmap.  
<br />
```c++
      //load_model
      for(int i=0;i<28;i++){
        for(int j=0;j<28;j++){
          input->data.f[i*28+j] = d[j][i];
        }  
      }
      // Run inference, and report any error
      TfLiteStatus invoke_status = interpreter->Invoke();
      if (invoke_status != kTfLiteOk) {
        MicroPrintf("Invoke failed");
        return;
      }
```
The model is invoked to run inference on the handwriting image by passing the image array into the input tensor. An error message is printed if this fails.  
<br />
```c++
      Serial.println();
      float out[26];
      for(int i=0;i<26;i++){
        out[i]=output->data.f[i];
        MicroPrintf("%c: %f ",i+65,out[i]);
      }
      float bestfitvalue;
      int numberrec;
      bestfitvalue=out[0];
      for(int i=0;i<26;i++){
        if(bestfitvalue<=out[i]){
          bestfitvalue=out[i];
          numberrec=i;
        }
      }
      MicroPrintf("Alphabet recognised: %c",numberrec+'A');
      tft.fillRect(351,120,20,20,TFT_WHITE);
      tft.setCursor(351, 120, 2);
      tft.printf("%c",numberrec+'A');
```
After inference is done, the logit values for all 26 alphabets are obtained from the output tensor and displayed. Then the highest logit value is determined and declared as the alphabet recognised by the model. The result is displayed on the screen and in the Serial Monitor.  
<br />
```c++
    //clicked reset button
    }else if(x>300&&x<350&&y>200&&y<250){
      for(int i=0;i<height;i++){
        for(int j=0;j<width;j++){
          d[i][j]=0;
        }
      }
      tft.fillRect(1,1,279,279,TFT_WHITE);
      tft.drawRect(0,0,281,281,TFT_RED);
    }
```
Finally, if the reset button(region) is touched then the drawing region is wiped clean and the image array is reset to all 0s.  
<br />
An example of the system working IRL:
![image](https://github.com/user-attachments/assets/5bff7603-28e0-44dc-84b9-b0c215a9d71a)  
As shown, the lines drawn are thicker compared to touch_test because of the 2x2 thickness “brush size”. It has a recognise alphabet button in green, rest button in red, and draw and erase button at the bottom. The handwriting of an S is correctly recognised.  
<br />
![image](https://github.com/user-attachments/assets/d302b2ce-c66e-4f80-982b-524a18933817)|![image](https://github.com/user-attachments/assets/f7a060af-9d36-45b1-b6a7-65135e470f59)  
The Serial Monitor displays the image and the logits for each alphabet, with the highest value being selected.
