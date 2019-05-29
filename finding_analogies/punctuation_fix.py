import re
import pandas as pd
from sys import argv
from fixes import fixes

"""
A lot of scanned text have extra spaced between punctuation signs, and this is fixing that.
Array containing regular expressions for common scanning errors and the corrected strings.
This should only be ran on analogy csvs that will be read by humans when needed, 
rather than running it on all files in the corpus.
"""

#TODO is there a way to fix quitation marks as well? 
#Some quation marks are supposed to be there, and does SHOULD NOT be removed.
in_path = argv[1]

#if the user does not give a second argument, the csv file is changed in place
if len(argv) == 3:
    out_path = argv[2]
else:
    out_path = in_path

df = pd.read_csv(in_path)

#fixes is a list if tuples with regular expressions matching errors and the corrected expressions.
def fix(x):
    for i in fixes:
        x = re.sub(i[0], i[1], str(x))
    return x


df['text'] = df['text'].apply(lambda x: fix(x))
df.to_csv(out_path, encoding='utf-8', index=False)

