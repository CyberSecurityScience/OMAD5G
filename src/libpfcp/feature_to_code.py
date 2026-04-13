
import re

START_LINE = r"^(\d+)\/(\d+)\s+(\w+)\s.*N4\s+(.*)"

state = 'none'

octet = 6
bit = 0
comment = ''
name = ''
with open('feature.txt', 'r') as fp :
    for l in fp :
        l = l.strip()
        if l :
            g = re.match(START_LINE, l)
            if g :
                if name and name != 'NULL' :
                    octet -= 9
                    # octet = 7 - octet
                    # pos = 8 * octet + bit - 1
                    octet = 3 - octet
                    pos = 8 * octet + bit - 1
                    #print(pos, name, comment)
                    print(f'\t/// {comment}')
                    print(f'\tpub get{name}, set{name}: {pos}, {pos};')
                octet = int(g.group(1))
                bit = int(g.group(2))
                name = g.group(3)
                comment = g.group(4)
            else :
                comment = comment + ' ' + l

