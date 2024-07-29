// Define pins for ultrasonic sensors
const int trigPin1 = 9;
const int echoPin1 = 10;
const int trigPin2 = 11;
const int echoPin2 = 12;

// Variables for distance measurement
long duration1, duration2;
int distance1, distance2;
int peopleCount = 0;
int peopleIn = 0;
int peopleOut = 0;

void setup() {
  // Set up the ultrasonic sensor pins
  pinMode(trigPin1, OUTPUT);
  pinMode(echoPin1, INPUT);
  pinMode(trigPin2, OUTPUT);
  pinMode(echoPin2, INPUT);

  // Start serial communication for debugging
  Serial.begin(115200);
}

void loop() {
  // Measure distance from sensor 1
  distance1 = measureDistance(trigPin1, echoPin1);
  // Measure distance from sensor 2
  distance2 = measureDistance(trigPin2, echoPin2);

  // Check for people entering (Sensor 1 detects first)
  if (distance1 < 50 && distance2 >= 50) {
    peopleCount++;
    peopleIn++;
    delay(5000); // Wait for the person to pass through
  }

  // Check for people exiting (Sensor 2 detects first)
  if (distance2 < 50 && distance1 >= 50) {
    peopleCount--;
    peopleOut++;
    delay(5000); // Wait for the person to pass through
  }

  // Print counts for debugging
  Serial.print("People In: ");
  Serial.print(peopleIn);
  Serial.print(", People Out: ");
  Serial.print(peopleOut);
  Serial.print(", Total: ");
  Serial.println(peopleCount);

  delay(200); // Small delay for stability
}

int measureDistance(int trigPin, int echoPin) {
  // Send a 10us pulse to trigger pin
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);

  // Read the echo pin, pulseIn returns the duration in microseconds
  long duration = pulseIn(echoPin, HIGH);

  // Calculate the distance in cm
  int distance = duration * 0.034 / 2;

  return distance;
}
