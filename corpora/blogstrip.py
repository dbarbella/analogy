# -*- coding: utf-8 -*-

import sys

file_name = sys.argv[1]
out_file_name = file_name + "_stripped.txt"

def content_line(line):
    return not (len(line_stripped) == 0 or line_stripped[0] == "<")
    
def no_urls(line):
    return line.replace("urlLink","")

file_handler = open(file_name, "r", encoding="utf-8")
out_file_handler = open(out_file_name, "w", encoding="utf-8")

for line in file_handler:
    # .decode('utf-8')
    line_stripped = line.lstrip()
    #print(line_stripped)
    if content_line(line_stripped):
        print(no_urls(line_stripped), file=out_file_handler)
        
    
