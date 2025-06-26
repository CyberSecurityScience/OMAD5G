import os
from pathlib import Path
from tqdm import tqdm

source = '/mnt/nvme0/android-malware-traffic-gen/CIC-AndMal2017/'
pcaps = [os.path.join(dp, f) for dp, dn, filenames in os.walk(source) for f in filenames if os.path.splitext(f)[1] == '.pcap']

def get_pcap_duration(pcap) :
    # run ./pcap_duration and convert out to float
    cmd = f'./pcap_duration \"{pcap}\"'
    duration = os.popen(cmd).read()
    duration = duration.strip()
    try :
        duration = float(duration)
    except ValueError :
        duration = 0.0
    return duration

with open('cic_with_length.txt', 'w') as f :
    for pcap in tqdm(pcaps) :
        filename = pcap
        duration = get_pcap_duration(pcap)
        f.write(f'{filename}\t{duration}\n')
