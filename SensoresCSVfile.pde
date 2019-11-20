import processing.serial.*;
Serial myPort; //creates a software serial port on which you will listen to Arduino
Table dataTable; //table where we will read in and store values. You can name it something more creative!

int numReadings = 100; //keeps track of how many readings you’d like to take before writing the file.
int readingCounter = 0; //counts each reading to compare to numReadings.

String fileName, val;
void setup()
{
  String portName = Serial.list()[0];
  //CAUTION: your Arduino port number is probably different! Mine happened to be 1. Use a “handshake” sketch to figure out and test which port number your Arduino is talking on. A “handshake” establishes that Arduino and Processing are listening/talking on the same port.
  //Here’s a link to a basic handshake tutorial: https://processing.org/tutorials/overview/

  myPort = new Serial(this, portName, 38400); //set up your port to listen to the serial port
  dataTable = new Table();

  dataTable.addColumn("id"); //This column stores a unique identifier for each record. We will just count up from 0 – so your first reading will be ID 0, your second will be ID 1, etc.

  //the following adds columns for time. You can also add milliseconds. See the Time/Date functions for Processing: https://www.processing.org/reference/
  dataTable.addColumn("year");
  dataTable.addColumn("month");
  dataTable.addColumn("day");
  dataTable.addColumn("hour");
  dataTable.addColumn("minute");
  dataTable.addColumn("second");

  //the following are dummy columns for each data value. Add as many columns as you have data values. Customize the names as needed. Make sure they are in the same order as the order that Arduino is sending them!
  dataTable.addColumn("FlexSensor");
  dataTable.addColumn("IMU2_X");
  dataTable.addColumn("IMU2_Y");
  dataTable.addColumn("IMU2_Z");
  dataTable.addColumn("IMU_X");
  dataTable.addColumn("IMU_Y");
  dataTable.addColumn("IMU_Z");
  //dataTable.addColumn("Class");
}

void serialEvent(Serial myPort) {
  try {
    val = myPort.readStringUntil('\n'); //The newline separator separates each Arduino loop. We will parse the data by each newline separator.
    if (val!= null) { //We have a reading! Record it.
      val = trim(val); //gets rid of any whitespace or Unicode nonbreakable space
      println(val); //Optional, useful for debugging. If you see this, you know data is being sent. Delete if you like.
      float sensorVals[] = float(split(val, ',')); //parses the packet from Arduino and places the valeus into the sensorVals array. I am assuming floats. Change the data type to match the datatype coming from Arduino.

      TableRow newRow = dataTable.addRow(); //add a row for this new reading
      newRow.setInt("id", dataTable.lastRowIndex());//record a unique identifier (the row’s index)

      //record time stamp
      newRow.setInt("year", year());
      newRow.setInt("month", month());
      newRow.setInt("day", day());
      newRow.setInt("hour", hour());
      newRow.setInt("minute", minute());
      newRow.setInt("second", second());

      //record sensor information. Customize the names so they match your sensor column names. 
      newRow.setFloat("FlexSensor", sensorVals[0]);
      newRow.setFloat("IMU2_X", sensorVals[1]);
      newRow.setFloat("IMU2_Y", sensorVals[2]);
      newRow.setFloat("IMU2_Z", sensorVals[3]);
      newRow.setFloat("IMU_X", sensorVals[4]);
      newRow.setFloat("IMU_Y", sensorVals[5]);
      newRow.setFloat("IMU_Z", sensorVals[6]);
      //newRow.setFloat("Class", 1);

      readingCounter++; //optional, use if you’d like to write your file every numReadings reading cycles

      //saves the table as a csv in the same folder as the sketch every numReadings.
      if (readingCounter % numReadings ==0)//The % is a modulus, a math operator that signifies remainder after division. The if statement checks if readingCounter is a multiple of numReadings (the remainder of readingCounter/numReadings is 0)
      {
        fileName = str(year()) + str(month()) + str(day()) + str(dataTable.lastRowIndex()); //this filename is of the form year+month+day+readingCounter
        saveTable(dataTable, fileName, "csv"); //Woo! save it to your computer. It is ready for all your spreadsheet dreams.
      }
    }
  }
  catch(RuntimeException e) {
    e.printStackTrace();
  }
}

void draw()
{
  //visualize your sensor data in real time here! In the future we hope to add some cool and useful graphic displays that can be tuned to different ranges of values.
}
