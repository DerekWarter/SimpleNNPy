import sys
import re
from unidecode import unidecode

# Standardizes wordlists by truncating them and finding ASCII equivalents for all Unicode chars.

fimport = open(sys.argv[1],"r")
fexport = open(sys.argv[2],"w")

i = 1
for l in fimport:

    if sys.argv[3] == "del":
        if i % 2 == 0:
            i += 1
            continue

    i += 1
    l = unidecode(l)
    goodtext = re.findall("^[a-zA-Z]+$", l.strip("\n"))
    if not goodtext:
        continue
    l = l.lower()
    if len(l) < 6:
        continue
    if len(l) > 15:
        l = l[:15] + "\n"
    fexport.write(l)
