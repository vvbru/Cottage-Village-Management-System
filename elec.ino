#include "Arduino.h"
#include "EspMQTTClient.h"

#define PUB_DELAY (30 * 1000) /* 30 seconds */

int t11 = 1200;
int t21 = 1000;
int t31 = 900;

int t12 = 500;
int t22 = 400;
int t32 = 250;

int t13 = 20;
int t23 = 10;
int t33 = 5;

int open1;
int open2;
int open3;

EspMQTTClient client(
  "OnePlus 8",
  "pwpr7155",

  "dev.rightech.io",
  "mqtt-vvburdyug-h4osdo"
);

void setup() {
  Serial.begin(9600);
}

void onConnectionEstablished() {

  Serial.println("connected");
  client.subscribe("base/relay/req", [] (const String & payload)  {
  Serial.println(payload);


  if (payload == "1") {publishEl("base/state/el1", t11, t21, t31, open1);}
  if (payload == "2") {publishEl("base/state/el2", t12, t22, t32, open2);}
  if (payload == "3") {publishEl("base/state/el3", t13, t23, t33, open3);}

  });
}


void publishEl(String topic, int t1, int t2, int t3, int open) {

  String info = String("{\"t1\": ")+
  String(t1)+String(", \"t2\":")+
  String(t2)+String(", \"t3\":")+
  String(t3)+String(", \"open\":")+
  String(open)+String("}");

  client.publish(topic, info);
}


long last = 0;
void loop() {

  t11 =  t11 + random(3);
  t21 =  t21 + random(3);
  t31 =  t31 + random(3);

  t12 =  t12 + random(3);
  t22 =  t22 + random(3);
  t32 =  t32 + random(3);

  t13 =  t13 + random(3);
  t23 =  t23 + random(3);
  t33 =  t33 + random(3);

  int ran = random(100);
  if (ran > 85) { open1 = 1; open2 = 0; open3 = 0; }
  if (ran > 90) { open1 = 0; open2 = 1; open3 = 0; }
  if (ran > 95) { open1 = 0; open2 = 0; open3 = 1; }

  client.loop();
  long now = millis();
  if (client.isConnected() && (now - last > PUB_DELAY)) {
     publishEl("base/state/el1", t11, t21, t31, open1);
     publishEl("base/state/el2", t12, t22, t32, open2);
     publishEl("base/state/el3", t13, t23, t33, open3);
     last = now;
  }


  delay(1000);
}
