#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tf_lite_model_AZ_handwriting.h"
#include "FS.h"
#include <SPI.h>
#include <TFT_eSPI.h>

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

// The name of this function is important for Arduino compatibility.
void setup() {
  uint16_t calibrationData[5];
  uint8_t calDataOK = 0;
  //initialise empty image array
  for(int i=0;i<height;i++){
    for(int j=0;j<width;j++){
      d[j][i]=0;
    }
  }
  Serial.begin(115200);
  Serial.println("starting");
  tft.init();
  tft.setRotation(3);
  tft.fillScreen((0xFFFF));
  
  //calibration run
  tft.setCursor(20, 0, 2);
  tft.setTextColor(TFT_BLACK, TFT_WHITE);  
  tft.setTextSize(1);
  tft.println("calibration run");

  // check file system
  if (!SPIFFS.begin()) {
    Serial.println("formatting file system");
    SPIFFS.format();
    SPIFFS.begin();
  }

  // check if calibration file exists
  if (SPIFFS.exists(CALIBRATION_FILE)) {
    File f = SPIFFS.open(CALIBRATION_FILE, "r");
    if (f) {
      if (f.readBytes((char *)calibrationData, 14) == 14)
        calDataOK = 1;
      f.close();
    }
  }
  if (calDataOK) {
    // calibration data valid
    tft.setTouch(calibrationData);
    tft.println("calibrated");
  } else {
    // data not valid. recalibrate
    tft.calibrateTouch(calibrationData, TFT_WHITE, TFT_RED, 15);
    // store data
    File f = SPIFFS.open(CALIBRATION_FILE, "w");
    if (f) {
      f.write((const unsigned char *)calibrationData, 14);
      f.close();
    }
  }
  //finish calibration
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
  
  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    MicroPrintf("AllocateTensors() failed");
    return;
  }

  // Obtain pointers to the model's input and output tensors.
  input = interpreter->input(0);
  output = interpreter->output(0);
  pinMode(redled,OUTPUT);
  pinMode(greenled,OUTPUT);
}

// The name of this function is important for Arduino compatibility.
void loop() {
  if(draw==1){
    digitalWrite(greenled,HIGH);
    digitalWrite(redled,LOW);
  }else{
    digitalWrite(greenled,LOW);
    digitalWrite(redled,HIGH);    
  }
  uint16_t x, y;//touch coordinates
  if (tft.getTouch(&x, &y)) {
    //tft.fillRect(300,60,100,100,TFT_WHITE);
    tft.setCursor(281, 5, 2);
    tft.printf("x: %i     ", x);
    tft.setCursor(281, 20, 2);
    tft.printf("y: %i    ", y);//display coordinates

    //draw button
    if(300<x && x<340 && 260<y && y<300){
      draw=1;

    //erase button
    }else if(340<x && x<380 && 260<y && y<300){
      draw=0;

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
  }
}
