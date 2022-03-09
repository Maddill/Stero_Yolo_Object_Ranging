import gpsd

gpsd.connect()

packet = gpsd.get_current()

print(packet.position)
