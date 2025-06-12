
# use scarpy 
from scapy.all import DNS, DNSQR, IP, sr1, UDP, Ether, srp, sendp, Raw
from scapy.contrib import gtp

#dns_req = IP(dst='8.8.8.8')/UDP(dport=53)/DNS(rd=1, qd=DNSQR(qname='www.thepacketgeek.com'))
dns_req = Ether(dst="1e:81:34:11:b1:7d",src="1e:81:34:11:b1:7d")/IP(dst="255.0.0.0",ttl=64)/UDP(dport=53)/DNS(rd=1, qd=DNSQR(qname='google.co.jp'))
# dns_req = Ether(dst="1e:81:34:11:b1:7d",src="1e:81:34:11:b1:7d")/IP(dst="255.0.0.0",ttl=64)/UDP(dport=99, sport=88)/Raw(load=('32 ff 00 58 00 00 00 01 '
# '28 db 00 00 45 00 00 54 00 00 40 00 40 00 5e a5 ca 0b 28 9e c0 a8 28 b2 08 00 '
# 'be e7 00 00 28 7b 04 11 20 4b f4 3d 0d 00 08 09 0a 0b 0c 0d 0e 0f 10 11 12 13 '
# '14 15 16 17 18 19 1a 1b 1c 1d 1e 1f 20 21 22 23 24 25 26 27 28 29 2a 2b 2c 2d '
# '2e 2f 30 31 32 33 34 35 36 37'))
answer = sendp(dns_req, iface='veth0', count=1)

