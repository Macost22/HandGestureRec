#include <Wire.h>

#include "SparkFun_BNO080_Arduino_Library.h"
#include "SparkFun_Displacement_Sensor_Arduino_Library.h"
#include "quaternionFilters.h"
#include "MPU9250.h"
#include <SoftwareSerial.h>
#define AHRS true        // Set to false for basic data read
#define SerialDebug true  // Set to true to get Serial output for debugging

int intPin = 12;  // These can be changed, 2 and 3 are the Arduinos ext int pins
int myLed  = 13;  // Set up pin 13 led for toggling

#define I2Cclock 400000
#define I2Cport Wire
//#define MPU9250_ADDRESS MPU9250_ADDRESS_AD0   // Use either this line or the next to select which I2C address your device is using
#define MPU9250_ADDRESS 0x68



MPU9250 myIMU(MPU9250_ADDRESS, I2Cport, I2Cclock);

BNO080 myIMU2;

ADS myFlexSensor; //Create instance of the Angular Displacement Sensor (ADS) class
float sensorVals[]= {0,0,0,0,0,0,0,0};
int flexSensorPin = A0;

void setup()
{
  Wire.begin();
  // TWBR = 12;  // 400 kbit/sec I2C speed
  Serial.begin(38400);

  myIMU2.begin();

  Wire.setClock(400000); //Increase I2C data rate to 400kHz

  myIMU2.enableRotationVector(50); //Send data update every 50ms

  //Serial.println(F("Accelerometer enabled"));
  //Serial.println(F("Output in form x, y, z, in m/s^2"));

  if (myFlexSensor.begin() == false)
  {
    //7Serial.println(F("No flex sensor detected. Check wiring. Freezing..."));
    while (1);
  }
  
  while(!Serial){};

  // Set up the interrupt pin, its set as active high, push-pull
  pinMode(intPin, INPUT);
  digitalWrite(intPin, LOW);
  pinMode(myLed, OUTPUT);
  digitalWrite(myLed, HIGH);

  // Read the WHO_AM_I register, this is a good test of communication
  byte c = myIMU.readByte(MPU9250_ADDRESS, WHO_AM_I_MPU9250);
  //Serial.print(F("MPU9250 I AM 0x"));
  //Serial.print(c, HEX);
  //Serial.print(F(" I should be 0x"));
  //Serial.println(0x71, HEX);

  if (c == 0x71) // WHO_AM_I should always be 0x71
  {
    //Serial.println(F("MPU9250 is online..."));

    // Start by performing self test and reporting values
    myIMU.MPU9250SelfTest(myIMU.selfTest);
    //Serial.print(F("x-axis self test: acceleration trim within : "));
    //Serial.print(myIMU.selfTest[0],1); Serial.println("% of factory value");
    //Serial.print(F("y-axis self test: acceleration trim within : "));
    //Serial.print(myIMU.selfTest[1],1); Serial.println("% of factory value");
    //Serial.print(F("z-axis self test: acceleration trim within : "));
    //Serial.print(myIMU.selfTest[2],1); Serial.println("% of factory value");
    //Serial.print(F("x-axis self test: gyration trim within : "));
    //Serial.print(myIMU.selfTest[3],1); Serial.println("% of factory value");
    //Serial.print(F("y-axis self test: gyration trim within : "));
    //Serial.print(myIMU.selfTest[4],1); Serial.println("% of factory value");
    //Serial.print(F("z-axis self test: gyration trim within : "));
    //Serial.print(myIMU.selfTest[5],1); Serial.println("% of factory value");

    // Calibrate gyro and accelerometers, load biases in bias registers
    myIMU.calibrateMPU9250(myIMU.gyroBias, myIMU.accelBias);

    myIMU.initMPU9250();
    // Initialize device for active mode read of acclerometer, gyroscope, and
    // temperature
    //Serial.println("MPU9250 initialized for active data mode....");

    // Read the WHO_AM_I register of the magnetometer, this is a good test of
    // communication
    byte d = myIMU.readByte(AK8963_ADDRESS, WHO_AM_I_AK8963);
    //Serial.print("AK8963 ");
    //Serial.print("I AM 0x");
    //Serial.print(d, HEX);
    //Serial.print(" I should be 0x");
    //Serial.println(0x48, HEX);

    if (d != 0x48)
    {
      // Communication failed, stop here
      //Serial.println(F("Communication failed, abort!"));
      Serial.flush();
      abort();
    }

    // Get magnetometer calibration from AK8963 ROM
    myIMU.initAK8963(myIMU.factoryMagCalibration);
    // Initialize device for active mode read of magnetometer
    //Serial.println("AK8963 initialized for active data mode....");

    //if (SerialDebug)
    //{
      //  Serial.println("Calibration values: ");
      //Serial.print("X-Axis factory sensitivity adjustment value ");
      //Serial.println(myIMU.factoryMagCalibration[0], 2);
      //Serial.print("Y-Axis factory sensitivity adjustment value ");
      //Serial.println(myIMU.factoryMagCalibration[1], 2);
      //Serial.print("Z-Axis factory sensitivity adjustment value ");
      //Serial.println(myIMU.factoryMagCalibration[2], 2);
    //}
    
    // Get sensor resolutions, only need to do this once
    myIMU.getAres();
    myIMU.getGres();
    myIMU.getMres();

    // The next call delays for 4 seconds, and then records about 15 seconds of
    // data to calculate bias and scale.
//    myIMU.magCalMPU9250(myIMU.magBias, myIMU.magScale);
    //Serial.println("AK8963 mag biases (mG)");
    //Serial.println(myIMU.magBias[0]);
    //Serial.println(myIMU.magBias[1]);
    //Serial.println(myIMU.magBias[2]);

    //Serial.println("AK8963 mag scale (mG)");
    //Serial.println(myIMU.magScale[0]);
    //Serial.println(myIMU.magScale[1]);
    //Serial.println(myIMU.magScale[2]);
//    delay(2000); // Add delay to see results before serial spew of data
//
//    if(SerialDebug)
//    {
//      Serial.println("Magnetometer:");
//      Serial.print("X-Axis sensitivity adjustment value ");
//      Serial.println(myIMU.factoryMagCalibration[0], 2);
//      Serial.print("Y-Axis sensitivity adjustment value ");
//      Serial.println(myIMU.factoryMagCalibration[1], 2);
//      Serial.print("Z-Axis sensitivity adjustment value ");
//      Serial.println(myIMU.factoryMagCalibration[2], 2);
//    }

  } // if (c == 0x71)
  else
  {
//    Serial.print("Could not connect to MPU9250: 0x");
//    Serial.println(c, HEX);

    // Communication failed, stop here
//    Serial.println(F("Communication failed, abort!"));
    Serial.flush();
    abort();
  }
}

void loop() 
{
  int flexSensorReading = analogRead(flexSensorPin);
  int flex0to90= map(flexSensorReading, 270, 470, 0, 90);
  if (myIMU2.dataAvailable() == true)
  {
    float quatI = myIMU2.getQuatI();
    float quatJ = myIMU2.getQuatJ();
    float quatK = myIMU2.getQuatK();
    float quatReal = myIMU2.getQuatReal();
    float quatRadianAccuracy = myIMU2.getQuatRadianAccuracy();

//    Serial.print("IMU 2:");
//    Serial.print(quatI, 2);
//    Serial.print(F(","));
//    Serial.print(quatJ, 2);
//    Serial.print(F(","));
//    Serial.print(quatK, 2);
//    Serial.print(F(","));
//    Serial.print(quatReal, 2);
//    Serial.print(F(","));
//    Serial.print(quatRadianAccuracy, 2);
//    Serial.print(F(","));

//    Serial.println();
  }
  // If intPin goes high, all data registers have new data
  // On interrupt, check if data ready interrupt
  if (myIMU.readByte(MPU9250_ADDRESS, INT_STATUS) & 0x01)
  {
    myIMU.readAccelData(myIMU.accelCount);  // Read the x/y/z adc values

    // Now we'll calculate the accleration value into actual g's
    // This depends on scale being set
    myIMU.ax = (float)myIMU.accelCount[0] * myIMU.aRes; // - myIMU.accelBias[0];
    myIMU.ay = (float)myIMU.accelCount[1] * myIMU.aRes; // - myIMU.accelBias[1];
    myIMU.az = (float)myIMU.accelCount[2] * myIMU.aRes; // - myIMU.accelBias[2];

    myIMU.readGyroData(myIMU.gyroCount);  // Read the x/y/z adc values

    // Calculate the gyro value into actual degrees per second
    // This depends on scale being set
    myIMU.gx = (float)myIMU.gyroCount[0] * myIMU.gRes;
    myIMU.gy = (float)myIMU.gyroCount[1] * myIMU.gRes;
    myIMU.gz = (float)myIMU.gyroCount[2] * myIMU.gRes;

    myIMU.readMagData(myIMU.magCount);  // Read the x/y/z adc values

    // Calculate the magnetometer values in milliGauss
    // Include factory calibration per data sheet and user environmental
    // corrections
    // Get actual magnetometer value, this depends on scale being set
    myIMU.mx = (float)myIMU.magCount[0] * myIMU.mRes
               * myIMU.factoryMagCalibration[0] - myIMU.magBias[0];
    myIMU.my = (float)myIMU.magCount[1] * myIMU.mRes
               * myIMU.factoryMagCalibration[1] - myIMU.magBias[1];
    myIMU.mz = (float)myIMU.magCount[2] * myIMU.mRes
               * myIMU.factoryMagCalibration[2] - myIMU.magBias[2];
  } // if (readByte(MPU9250_ADDRESS, INT_STATUS) & 0x01)

  // Must be called before updating quaternions!
  myIMU.updateTime();

  // Sensors x (y)-axis of the accelerometer is aligned with the y (x)-axis of
  // the magnetometer; the magnetometer z-axis (+ down) is opposite to z-axis
  // (+ up) of accelerometer and gyro! We have to make some allowance for this
  // orientationmismatch in feeding the output to the quaternion filter. For the
  // MPU-9250, we have chosen a magnetic rotation that keeps the sensor forward
  // along the x-axis just like in the LSM9DS0 sensor. This rotation can be
  // modified to allow any convenient orientation convention. This is ok by
  // aircraft orientation standards! Pass gyro rate as rad/s
  MahonyQuaternionUpdate(myIMU.ax, myIMU.ay, myIMU.az, myIMU.gx * DEG_TO_RAD,
                         myIMU.gy * DEG_TO_RAD, myIMU.gz * DEG_TO_RAD, myIMU.my,
                         myIMU.mx, myIMU.mz, myIMU.deltat);

  if (!AHRS)
  {
    myIMU.delt_t = millis() - myIMU.count;
    if (myIMU.delt_t > 500)
    {
//      if(SerialDebug)
//      {
//        // Print acceleration values in milligs!
//        Serial.print("X-acceleration: "); Serial.print(1000 * myIMU.ax);
//        Serial.print(" mg ");
//        Serial.print("Y-acceleration: "); Serial.print(1000 * myIMU.ay);
//        Serial.print(" mg ");
//        Serial.print("Z-acceleration: "); Serial.print(1000 * myIMU.az);
//        Serial.println(" mg ");
//        
//        // Print gyro values in degree/sec
//        Serial.print("X-gyro rate: "); Serial.print(myIMU.gx, 3);
//        Serial.print(" degrees/sec ");
//        Serial.print("Y-gyro rate: "); Serial.print(myIMU.gy, 3);
//        Serial.print(" degrees/sec ");
//        Serial.print("Z-gyro rate: "); Serial.print(myIMU.gz, 3);
//        Serial.println(" degrees/sec");
//      }

      myIMU.count = millis();
      digitalWrite(myLed, !digitalRead(myLed));  // toggle led
    } // if (myIMU.delt_t > 500)
  } // if (!AHRS)
  else
  {
    // Serial print and/or display at 0.5 s rate independent of data rates
    myIMU.delt_t = millis() - myIMU.count;

    // update LCD once per half-second independent of read rate
    if (myIMU.delt_t > 500)
    {
//      if(SerialDebug)
//      {
//        Serial.print("ax = ");  Serial.print((int)1000 * myIMU.ax);
//        Serial.print(" ay = "); Serial.print((int)1000 * myIMU.ay);
//        Serial.print(" az = "); Serial.print((int)1000 * myIMU.az);
//        Serial.println(" mg");
//
//        Serial.print("gx = ");  Serial.print(myIMU.gx, 2);
//        Serial.print(" gy = "); Serial.print(myIMU.gy, 2);
//        Serial.print(" gz = "); Serial.print(myIMU.gz, 2);
//        Serial.println(" deg/s");
//
//        Serial.print("mx = ");  Serial.print((int)myIMU.mx);
//        Serial.print(" my = "); Serial.print((int)myIMU.my);
//        Serial.print(" mz = "); Serial.print((int)myIMU.mz);
//        Serial.println(" mG");
//
//        Serial.print("q0 = ");  Serial.print(*getQ());
//        Serial.print(" qx = "); Serial.print(*(getQ() + 1));
//        Serial.print(" qy = "); Serial.print(*(getQ() + 2));
//        Serial.print(" qz = "); Serial.println(*(getQ() + 3));
//      }

      myIMU.yaw   = atan2(2.0f * (*(getQ()+1) * *(getQ()+2) + *getQ()
                    * *(getQ()+3)), *getQ() * *getQ() + *(getQ()+1)
                    * *(getQ()+1) - *(getQ()+2) * *(getQ()+2) - *(getQ()+3)
                    * *(getQ()+3));
      myIMU.pitch = -asin(2.0f * (*(getQ()+1) * *(getQ()+3) - *getQ()
                    * *(getQ()+2)));
      myIMU.roll  = atan2(2.0f * (*getQ() * *(getQ()+1) + *(getQ()+2)
                    * *(getQ()+3)), *getQ() * *getQ() - *(getQ()+1)
                    * *(getQ()+1) - *(getQ()+2) * *(getQ()+2) + *(getQ()+3)
                    * *(getQ()+3));
      myIMU.pitch *= RAD_TO_DEG;
      myIMU.yaw   *= RAD_TO_DEG;

      // Declination of SparkFun Electronics (40°05'26.6"N 105°11'05.9"W) is
      //   8° 30' E  ± 0° 21' (or 8.5°) on 2016-07-19
      // - http://www.ngdc.noaa.gov/geomag-web/#declination
      myIMU.yaw  -= 8.5;
      myIMU.roll *= RAD_TO_DEG;

//      if(SerialDebug)
//      {
//        Serial.print("Yaw, Pitch, Roll: ");
//        Serial.print(myIMU.yaw, 2);
//        Serial.print(", ");
//        Serial.print(myIMU.pitch, 2);
//        Serial.print(", ");
//        Serial.println(myIMU.roll, 2);
//
//        Serial.print("rate = ");
//        Serial.print((float)myIMU.sumCount / myIMU.sum, 2);
//        Serial.println(" Hz");
//      }

      myIMU.count = millis();
      myIMU.sumCount = 0;
      myIMU.sum = 0;
    } // if (myIMU.delt_t > 500)
  } // if (AHRS)

  
  if (myFlexSensor.available() == true)
  {
    sensorVals[0]= myFlexSensor.getX();
    //Serial.print(myFlexSensor.getX());
    //Serial.println();
  }
  else{
    sensorVals[0] = -9999;
  }

  //delay(10);
  //sensorVals[0]= myFlexSensor.getX();
  sensorVals[1]= -(flex0to90-89);
  sensorVals[2]= myIMU2.getQuatI();
  sensorVals[3]= myIMU2.getQuatJ();
  sensorVals[4]= myIMU2.getQuatK();
  sensorVals[5]= myIMU.yaw;
  sensorVals[6]= myIMU.pitch;
  sensorVals[7]= myIMU.roll;
  
  Serial.print(sensorVals[0]);
  Serial.print(",");
  Serial.print(sensorVals[1]);
  Serial.print(",");
  Serial.print(sensorVals[2]);
  Serial.print(",");
  Serial.print(sensorVals[3]);
  Serial.print(",");
  Serial.print(sensorVals[4]);
  Serial.print(",");
  Serial.print(sensorVals[5]);
  Serial.print(",");
  Serial.print(sensorVals[6]);
  Serial.print(",");
  Serial.print(sensorVals[7]);
  Serial.print(",");
  Serial.println("1");
  delay(1000);
  
  
}
