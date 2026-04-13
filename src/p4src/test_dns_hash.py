
from crccheck.crc import CrcArc
import math
import zlib

# Closest power of 2
def cl_p2(n):
    lg2 = int(math.log2(n))
    return 2 ** lg2



def crc32(domain: str, variant='32', reverse = False):
    max_bytes_per_label = 31
    max_bytes = max_bytes_per_label
    closest_power2 = cl_p2(max_bytes)
    sum_reg = b''
    sum_16 = b''
    splitdomains = reversed(domain.split('.')) if reverse else domain.split('.')
    for label in splitdomains:
        i = closest_power2
        temp_label = label
        while i >= 1:
            if len(temp_label) >= i:
                if i >= 16:
                    sum_16 += temp_label[:i].encode('ascii')   
                else:
                    sum_reg += temp_label[:i].encode('ascii')   
                temp_label = temp_label[i:]
            else:
                if i >= 16:
                    sum_16 += (0).to_bytes(i, byteorder='big')
                else:
                    sum_reg += (0).to_bytes(i, byteorder='big')
            i = int(i/2)
    total = (CrcArc.calc(sum_reg) if variant == '16' else zlib.crc32(sum_reg))
    if max_bytes >= 16:
        return (total + (CrcArc.calc(sum_16) if variant == '16' else zlib.crc32(sum_16))) % (2 ** int(variant))
    return total

for d in ['google.co.jp', 'google.com'] :
    #print(hex(crc32(d, reverse=True)))
    print(d, hex(crc32(d, reverse=False)))
