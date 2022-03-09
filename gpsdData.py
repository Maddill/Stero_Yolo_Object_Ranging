from serial import *
from pyubx2 import *

stream = Serial('COM11', 9600, timeout=3)
ubr = UBXReader(stream)
(raw_data, parsed_data) = ubr.read()
msg = UBXReader.parse(b"\xb5b\x13\x40\x14\x00\x01\x00\x01\x02\x01\x02\x03\x04\x01\x02\x03\x04\x01\x02\x03\x04\x01\x02\x03\x04\x93\xc8", msgmode=SET)
# msg = UBXReader.parse(message=)
print(raw_data)
print(parsed_data)

# from serial import Serial
# from pyubx2 import UBXMessage, SET

# serialOut = Serial('COM12', 9600, timeout=5)
# msg = UBXMessage('CFG','CFG-MSG', SET, msgClass=0xf0, msgID=0x01, rateUART1=1, rateUSB=1)
# print(msg)
# output = msg.serialize()
# serialOut.write(output)