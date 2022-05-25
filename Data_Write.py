import time
import pandas as pd
import datetime

from paho.mqtt import client as mqtt_client


broker = 'dev.rightech.io'
port = 1883

client_id = 'mqtt-vvburdyug-so188i'
# username = 'user'
# password = 'pass'

def connect_mqtt():
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("Connected to MQTT Broker!")
        else:
            print("Failed to connect, return code %d\n", rc)

    client = mqtt_client.Client(client_id)
    client.on_connect = on_connect
    client.connect(broker, port)
    return client


def subscribe(client: mqtt_client):
    def on_message(client, userdata, msg):

        plate = msg.payload.decode().split(',')[0]
        owner = msg.payload.decode().split(',')[1]
        d = msg.payload.decode().split(',')[2]

        year = d.split('.')[2]
        month = d.split('.')[1]
        day = d.split('.')[0]

        df = pd.read_csv("data.csv")

        if plate in df['Номер'].values:
            df.loc[df['Номер'] == plate,'Участок'] = owner
            df.loc[df['Номер'] == plate,'Год'] = year
            df.loc[df['Номер'] == plate,'Месяц'] = month
            df.loc[df['Номер'] == plate,'День'] = day
            client.publish('base/state/msg1', 'Данные обновлены')
        else:
            nreg = {'Номер':plate ,'Участок':owner,'Год':year,'Месяц':month,'День':day}
            df = df.append(nreg, ignore_index=True)
            client.publish('base/state/msg1', 'Данные добавлены')

        df.to_csv("data.csv", index=False)

    client.subscribe("base/data/plateinfo")
    client.on_message = on_message


def run():
    client = connect_mqtt()
    subscribe(client)
    client.loop_forever()

if __name__ == '__main__':
    run()
