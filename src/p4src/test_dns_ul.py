from scapy.all import DNS, DNSQR, DNSRR, IP, UDP, Ether, sendp, Raw
from scapy.contrib.gtp import  GTPHeader, GTPPDUSessionContainer

# Construct the packet: outer Ethernet/IP/UDP (GTP port 2152) carrying a GTP header and container,
# which encapsulates the inner IP/UDP/DNS packet.
for idx in range(4) :
    idx += 100
    q = "aaxx.google.com"
    dns_req = (
        Ether(dst="1e:81:34:11:b1:8d", src="1e:81:34:11:b1:8d") /
        IP(src="127.0.0.1", dst="10.99.0.1", ttl=64) /
        UDP(dport=2152) /
        # GTP-U header with a sample TEID and message type (0xff for G-PDU)
        GTPHeader(teid=1, version=1, gtp_type=0xff, E=1, next_ex=133) /
        # The GTP PDU Session Container; you can add additional fields if needed
        GTPPDUSessionContainer(type=1,QFI=1) /
        # The inner payload: an IP/UDP packet carrying a DNS request
        IP(src="10.10.0.2", dst="8.8.8.8", ttl=64) /
        UDP(dport=53) /
        DNS(id=idx,rd=1, qd=DNSQR(qname=q))
    )
    local_ip = '1.2.3.4'

    # Send the packet on interface 'veth0'
    sendp(dns_req, iface="veth0", count=1)

    #dns_req = IP(dst='8.8.8.8')/UDP(dport=53)/DNS(rd=1, qd=DNSQR(qname='www.thepacketgeek.com'))
    dns_req = Ether(dst="1e:81:34:11:b1:7d",src="1e:81:34:11:b1:7d")/IP(dst="10.10.0.2",src='8.8.8.8',ttl=64)/UDP(dport=53)/DNS(
        id=idx,
        qd=DNSQR(qname=q),
        aa=1,
        rd=0,
        qr=1,
        qdcount=1,
        ancount=1,
        nscount=0,
        arcount=0,
        ar=DNSRR(
            rrname=q,
            type='A',
            ttl=600,
            rdata=local_ip)
        )
    sendp(dns_req, iface='veth0', count=1)



def get_hashed_domain(domain):
    parts = domain.split('.')
    parts = parts[:4]
    if len(parts) <= 3 :
        return '.'.join(parts)
    parts = parts[-3:]
    return '.'.join(parts)

