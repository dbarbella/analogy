import re
import pandas as pd
from sys import argv
from fixes import fixes


in_path = argv[1]

if len(argv) == 3:
    out_path = argv[2]
else:
    out_path = in_path

df = pd.read_csv(in_path)

def fix(x):
    for i in fixes:
        x = re.sub(i[0], i[1], str(x))
    return x


df['text'] = df['text'].apply(lambda x: fix(x))
df.to_csv(out_path, encoding='utf-8', index=False)

