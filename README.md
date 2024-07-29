Documentation of esp32 handwriting recognition system

Here are details of the step by step guide to creating a handwriting recognition system utilizing ESP32 microcontroller, machine learning with Tensorflow, touch screen input, and Arduino programming. Items required are listed below:

-   ESP32 dev kit (with GPIO pins)
-   ILI9488 TFT Module
-   Breadboard
-   Jumper wires
-   Micro USB Type-B to USB Type-A cable
-   Computer with USB Type-A socket
-   Red and Green LEDs (optional)
-   Resistors (optional)

Handwriting system

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

Uncomment this line (line54):

```c++
#define ILI9488_DRIVER     // WARNING: Do not connect ILI9488 display SDO to MISO if other devices share the SPI bus (TFT SDO does NOT tristate when CS is high)
```

Comment out these lines (line170-176):

```c++
//#define TFT_MISO  PIN_D6  // Automatically assigned with ESP8266 if not defined
//#define TFT_MOSI  PIN_D7  // Automatically assigned with ESP8266 if not defined
//#define TFT_SCLK  PIN_D5  // Automatically assigned with ESP8266 if not defined

//#define TFT_CS    PIN_D8  // Chip select control pin D8
//#define TFT_DC    PIN_D3  // Data Command control pin
//#define TFT_RST   PIN_D4  // Reset pin (could connect to NodeMCU RST, see next line)
```

Uncomment this line (line230):

```c++
#define TOUCH_CS 21     // Chip select pin (T_CS) of touch screen
```

Go to User_Setup_Select.h and uncomment this line (line52):

```c++
#include <User_Setups/Setup21_ILI9488.h>           // Setup file for ESP32 and ILI9488 SPI bus TFT
```

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

Save all the changes made and return back to the Arduino IDE.

Now that the user setup for the library has been properly configured for our hardware, we can begin coding for the touchscreen functionality. Go to examples and under TFT_eSPI-\>480 x 320 open Touch_Controller_Demo. This example sketch will be the starting point and template for touchscreen controls. The Touch_Controller_Demo consists of 2 parts, calibrating the screen and drawing with touch input. The code in the void setup() portion checks for calibration values within the ESP32 SPIFFS internal storage and uses it to calibrate the screen touch sensors. If there is no calibration value available, the program starts a calibration run on the screen that prompts the user to touch the 4 corners of the screen to calibrate. The void loop() section then takes user touch input on the screen and calculates the coordinates touched. The program draws a pixel at the coordinate and this runs continuously.

Before testing this sketch, we will have to setup the hardware. Place the ESP32 pins between 2 breadboards and the ILI9488 screen pins separate from the ESP32. Using jumper cables, connect the screen VCC pin to ESP32 3.3V output. Connect the GND pin on the screen to the GND pin on the ESP32. The CS pin is connected to GPIO 15 pin, RESET pin is connected to GPIO 4 pin, DC/RS pin is connected to GPIO 2 pin, SDI(MOSI) pin is connected to GPIO 23 pin, SCK pin is connected to GPIO 18 pin, LED pin is connected to 3.3V pin, T_CLK pin is connected to GPIO 18 pin, T_CS pin is connected to GPIO 21, T_DIN pin is connected to GPIO 23 pin, T_DO pin is connected to GPIO 19 pin. Do not connect SDO(MOSI) pin and T_IRQ pin to anything. The GPIO pins numbers can be changed as long as it is also changed within Setup21_ILI9488.h and User_Setup.h to the corresponding values. After all the connections have been completed, plug in the micro USB type-B cable into the ESP32 and connect it to a USB port on the computer. Make sure the port section under tools is available and selected (example COM 5). Enable Erase All Flash Before Sketch Upload option to ensure there is no data from previous uses interfering. Verify and upload the example sketch to the ESP32 and check whether it is working as expected. The screen should light up and prompt the user to perform calibration as it is the first time running and have no calibration data. Tap the 4 red boxes at the corners to complete calibration and then begin drawing on the screen with a pointy object such as a plastic pen or finger. It should leave behind multicoloured pixels wherever it is touched. This confirms the display and touch functionality is working. Notice the x and y coordinates displayed on the screen, it represents where on the screen is touched and ranges between 0 to 320 on the y-axis and 0 to 480 on the x-axis. If the coordinate values do not match the area touched, then it might mean the calibration is off and needs to be recalibrated by reuploading the sketch with Enable Erase All Flash Before Sketch Upload option enabled. Once the calibration is accurate then this option can be disabled for future uploads to avoid recalibration.

![image](https://github.com/user-attachments/assets/e6ef7113-05de-4bc3-8774-90f0eb2c8ee5)

One thing to notice is the trails left behind is rather thin and faint, which is not useful for handwriting recognition. In order for the handwriting to be usable, it must be converted into a 28x28 bitmap. Rather than working with a 28x28 pixel area, the drawing region will be 280x280 pixels and subdivided into 10x10 pixels chunks as the pixels for handwriting bitmap. The code for the current sketch will have to be modified for the new requirements. Go to the link below and open touch_test/touch_test.ino:

<https://github.com/laibw/Handwriting-recognition>

The difference between this code and the previous code is that a 28x28 2D array is initialised at the beginning of the sketch which will be used to store our bitmap. The array is initialised to all 0s to represent an empty image. At the end of void setup() a red rectangular frame is drawn to represent the drawing area.

![A computer code with text Description automatically generated with medium confidence](media/2adb21ce22a2288ffadaa28746c79b86.png)

This bit of code in void loop() purpose is to calculate which 10x10 subdivision is being touched and filling in the particular spot in the array. Example: if position is (250,200), then the array element [25][20] is set to 1. The algorithm takes advantage of the rounding done when dividing integers such that the remainder/decimal is ignored and the number is rounded down, so position 0-9 is counted as position 0, 10-19 is position 1 etc. After filling in the array, the area on the screen is also filled in with a 10x10 black box as to represent the drawn pixel, with the position of the box determined by the coordinates touched. If the region touched is not within the drawing region then the array(bitmap) info is sent to the Serial Monitor where we can confirm its validity.

![](media/df52de398b5dd91155d557f54ca388cf.jpeg)![A grid of numbers Description automatically generated](media/73ed6939da674c654c8544b05098fe02.png)

The user should be able to produce images similar to the above by drawing within the red box region, and after clicking outside the box would produce the image on the right in the Serial Monitor. Storing the image as an array will be the main method of representing the handwriting, which can then be used as input on neural networks for handwriting recognition.

Training neural network model for handwriting recognition

The target of our handwriting recognition system is to be able to recognise handwritten alphabets. This will require a neural network that has been trained on handwritten alphabets to detect the users handwriting and output the closest matched alphabet. In order to train the model, high quality training data with labels is required. The link below contains a downloadable csv file that contains 300000+ examples of handwritten alphabets in grayscale format with labels.

[A-Z Handwritten Alphabets in .csv format (kaggle.com)](https://www.kaggle.com/datasets/sachinpatel21/az-handwritten-alphabets-in-csv-format)

Save the download into google drive as it will be useful later.

Open Google Colab (or Jupyter notebook on local machine, recommended using colab for computation resources) and open Modeltraining.ipynb file below:

[Handwriting-recognition/Modeltraining.ipynb at main · laibw/Handwriting-recognition (github.com)](https://github.com/laibw/Handwriting-recognition/blob/main/Modeltraining.ipynb)

This is a python notebook which contains code to train the handwriting recognition model with Tensorflow/Tensorflowlite. Tensorflow is a machine learning framework that helps with simplifying machine learning processes, while Tensorflowlite is a lightweight version of Tensorflow suitable for edge and embedded devices like ESP32. The notebook is spilt into code blocks that will be run sequentially and separately for easier understanding.

Starting with the first code block:

**![A screenshot of a computer code Description automatically generated](media/032e35d62d06c8e135f082c66e3c68c2.png)**

This imports all the library packages that will be used and includes useful packages like Tensorflow, Numpy etc.

![](media/98274cbe974b3f4ca0d01430171f32cf.png)

This line is used to map output numbers to an alphabet which will be useful later.

![A close-up of a computer code Description automatically generated](media/b1a31bde1a1c331d752b8fcd881a9897.png)

This defines a function to calculate file size.

![](media/6158399b742ae2a03d8acf338605fd53.png)

This line is for mounting google drive onto google colab environment so that files in google drive can be transferred over to colab. (Manually transferring files from local machine to google colab is very slow, google drive is much faster).

![A white card with black text Description automatically generated](media/febd36aaaab63dd4739da73d843b169d.png)

After running this line, a prompt will show up asking for permission to access google drive. Follow through all the steps and connect google colab to the google drive account which contains the csv data.

![A screenshot of a computer Description automatically generated](media/53a29acc50e5441b3ed5377c655bf451.png)

Here the data from the csv file in google drive is read and stored in variable df (Note that the csv file name here is handwritten_data_785.csv, but it could be different and has to be changed in the code accordingly). df.shape shows the shape of the df: (372037, 785). 372037 examples of alphabet handwriting and 784 pixel value +1 label value.

![A black and white image of a mathematical equation Description automatically generated](media/4f59abcb5582e2251162a7be1cee211a.png)

This splits the data into pixel values(features) and labels using array slicing. [:,1:] means all rows and second to last column, [:,0] means all rows and first column (since the label value is stored in the first column).

![A close up of a text Description automatically generated](media/1ef6ce2721e2ad850e327a9e136b83d4.png)

This sections deals with preprocessing the data to be more suitable for training. First the features are divided by 255 to convert the range from 0-255 to 0-1 which will be easier to train. The pixel values are then converted from grayscale to binary using np.ceil() function which makes all non 0 pixels 1. This is done to match the input method of the handwriting system as a binary bitmap, which will produce a model that is better suited for the current usage. The labels currently are in the form of 0-25 to represent alphabets, but it will be more useful if it’s a one-hot encoded array with all elements as 0 except the position corresponding to the label itself as 1. This is done using the keras.utils.to_categorical() function. A label that was originally 13 for example will become a 1D array of all 0 except index 13 which is 1.

![](media/c829ec2c2592e22fd2acdbf50041af6d.png)

This line splits the labels and features into training and testing datasets randomly in a 75:25 split. 42 is the random seed used so that the dataset can be replicated exactly between examples. x are features and y are labels. Splitting the dataset is important to avoid overfitting the training data set during training.

![A close-up of a computer code Description automatically generated](media/dbcb3dcb94336f85549f87d30bf64cf6.png)

Prints out the shape of the dataset. x_train should have 279027 rows and 784 columns(28x28) while x_test have 93010 rows and the same amount of columns. y_train and y_test should have the same amount of rows are the x counterparts and 26 columns representing the one-hot array.

![A computer code with black text Description automatically generated with medium confidence](media/2c4b333ca7599b9102b8d53d64dc122d.png)

This block of code defines the model layers that will be implemented. Dense means a fully interconnected layer of nodes with the first parameter being the number of nodes. The activation function used is relu(rectified linear) which is a commonly used activation function for machine learning. The input shape is the number of features that is used as input, in this case 784 pixels. The model here is a simple sequential neural network with 2 layers having 128 and 26 nodes respectively. The second layer has the same amount of nodes as the output(26 alphabets). The output node with the highest activation will be considered as the predicted alphabet.

![A screenshot of a computer Description automatically generated](media/a7988fc095f11f3ad2652c6e0f1cd18d.png)

The number of parameters can be determined by the number of interconnections between nodes as these connections holds the model weights(parameters). Total parameters=(784+1(bias))x128+(128+1(bias))x26=103834 which matches the summary.

![A black text with red and black text Description automatically generated](media/3af90e3cd2c53979b0f8f1aa88211adf.png)

After defining the model structure, the model is set up for training using model.compile(). The optimizer is set to adam, and the loss function by which the model is trained to minimise is CategoricalCrossentropy with from_logits set to true because the output values have not undergone softmax and are considered logits. A metric to pay attention to is accuracy which is the rate of a correct prediction.

![A black text with purple lines Description automatically generated](media/f3b391cd878b481be93dac9916dfc6f7.png)

This block is run to delete redundant data to save memory (colab offers only 12GB on free version). If memory is not enough, the runtime will crash, and training cannot complete.

![](media/a37730b1b34dd0f738f59a5ac0224c55.png)

This line will begin running the training algorithm for 10 epochs(cycles) by tuning the weights based on minimizing the loss function. The loss value will decrease as training goes on and accuracy will increase.

![A close-up of a number Description automatically generated](media/42d9983d3098e1ff3d1f7a56354be486.png)

This is to evaluate the model against a separate testing dataset that the model has never seen before. If the accuracy does not differ significantly from training accuracy then the model is considered to have not overfitted. In this case the model achieve over 96% accuracy on the testing dataset.

![A close-up of a computer code Description automatically generated](media/53ce3a86fa95a0b0f70df2e0a6a6a815.png)

Now save the model as a backup. The model can be found under the content folder in google colab.

![A close-up of a computer screen Description automatically generated](media/c5524f67820d8e9c223c821916321658.png)

This shows the size of the model, in this case 1.2MB which is too big for ESP32. The model has to be shrunk in order to deploy on the microcontroller’s limited memory.

![A computer screen shot of a program code Description automatically generated](media/aede2629c60a6fa1b2dbbaa01fe14831.png)

A piece of code to demonstrate how the model works. By taking a few sample features from the testing dataset and running through the model, the model will output different an array of 26 values representing how close the handwriting is to each alphabet. The highest value position is then translated into the corresponding alphabet as the chosen predicted alphabet.

![A close up of a letter Description automatically generated](media/1c23ef3be2c9c2a401363b42dba2d2ac.png)

As shown the model can correctly predict the samples tested.

In order to run the model on ESP32, the model has to be smaller to fit into the limited memory. As good way to do this is to convert the Tensorflow model into a Tensorflowlite model. Tensorflowlite does a few optimizations and quantization to reduce the space required to store the parameter(weights) which the model is mainly comprised of.

![A screen shot of a computer code Description automatically generated](media/38e64368f5bfcb6876794a4b1a63679f.png)

This block of code is to convert the Tensorflow model into a tensorflowlite model. A converter is set up that takes in the tensorflow model and applies various optimizations to reduce the size of the model. The main optimization used is quantization which converts the weights(which are usually stored as floats) into a smaller data type without losing much accuracy. In order to apply quantization a representative dataset should be provided for the algorithm to estimate and calibrate the range weights so that it can optimize the size. The representative dataset is taken from the test dataset in this case.

![](media/427d3f69c93741ca62ecefeada14cce1.png)

This line saves the tensorflowlite model as a tflite file which can be found under the content folder of google colab. The output shows the size of the converted model which is around 100kB which is an order of magnitude smaller and more suitable to be run on the ESP32.

![](media/ae4ea66d137d5d0dfaea0896c32df3aa.png)

To test the tensorflowlite version of the model, it has to be run through an interpreter. The input tensor(array) accepts only 1 sample at a time as can be seen in the input_details shape. To test a large scale dataset all at once the input tensor has to be reshaped to accept more samples at once.

![A screen shot of a computer code Description automatically generated](media/977dc249a0a4dedd2e8cfb64bddd6b9b.png)

This block is responsible for reshaping the input tensor to take in multiple samples at once. The input tensor now has 93010 rows, same as the number of test samples.

![](media/85d483c3168ceca33e1e94aec8dd0442.png)

Prepare a testing dataset that has a compatible datatype with the input tensor(np.float32).

![A screenshot of a computer code Description automatically generated](media/f70c40775ff85afadf0404bd8c61c921.png)

This code invokes the interpreter to perform inference on the input data and then takes the predicted values from the output tensor to compare against the actual values. As can be seen, the accuracy is about 96% which is similar to the tensorflow model and has minimal loss of accuracy from conversion. This is desirable and proves that tensorflowlite models are viable to be used.

![](media/73689421c985e0a255b57e0d734b59f5.png)

Finally, convert the tflite model into a C++ file by using xxd command to dump the hex data into a c array. This is done to more easily include the model in the Arduino sketch. Download the cc file on the local machine.

After this the training phase is done and integration of handwriting system and handwriting recognition model can be performed.
