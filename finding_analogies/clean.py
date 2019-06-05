from get_path import get_path
from subprocess import call, Popen
from get_path import get_path
from sys import argv

#cleans files for processing using clean.sh
#is called from gutenger_scraper.py, but can also be called manually if that fails.
def clean(begin,end,output_dir):
    Popen(["bash", "-c", "chmod +x text_cleaner.sh"])
    for i in range(begin,end+1,100):
        clean_path = get_path(i,output_dir)
        call(["bash","./text_cleaner.sh", clean_path])
        print("cleaned", clean_path)

if __name__ == "__main__":
    begin = int(argv[1])
    end = argv[2]
    if end == "all":
        end = 59000
    else:
        end = int(end)
    output_dir = argv[3]
    if output_dir[-1] != "/":
        output_dir += "/"
    clean(begin,end, output_dir)
