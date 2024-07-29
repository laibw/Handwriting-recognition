#include "FS.h"
#include <SPI.h>
#include <TFT_eSPI.h>

TFT_eSPI tft = TFT_eSPI();

#define CALIBRATION_FILE "/calibrationData"
const int width = 28;
const int height = 28;
int d[width][height];

void setup(void) {
  uint16_t calibrationData[5];
  uint8_t calDataOK = 0;
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

  tft.fillScreen((0xFFFF));
  tft.drawRect(0,0,280,280,TFT_RED);
}

void loop() {
  uint16_t x, y;

  if (tft.getTouch(&x, &y)) {

    tft.setCursor(281, 5, 2);
    tft.printf("x: %i     ", x);
    tft.setCursor(281, 20, 2);
    tft.printf("y: %i    ", y);
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
  }
}
