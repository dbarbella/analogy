#!/usr/bin/env python
import requests
from bs4 import BeautifulSoup
from multiprocessing import Pool
from sys import argv
from clean import clean
from subprocess import call, Popen
import os
from time import sleep
from get_path import get_path

"""
This is to scrap the all the English books on gutenberg.org as of 04/15/2019
The output will be a number of .txt files
The output directory should be an argument in the command line
It's ran as:
python gutenberg_scrapper.py output_dir begin_num end_num
"""
#directory where the text files will be saved
#adds a [/] to the end in case the user forgot, otherwise the text files will end up in the
#directory the script is run in, but named as outputdirbook_id. If you want them to run in the directory
#the script is ran, you can set output_dir to ./
output_dir = argv[1]
if output_dir[-1] != "/":
    output_dir += "/"
    



#the number of books changes over time, there could be more in the future
NUM_BOOKS = 59000
URL = "https://www.gutenberg.org/ebooks/"

#lowest book number to be scraped
begin = int(argv[2])
#highest book number to be scraped
end = argv[3]

#the user can type in all as an argument to srape all the books available.
if end == "all":
    end = NUM_BOOKS
else:
    end = int(end)


#each english book is a url "https://www.gutenberg.org/ebooks/"
#followed by a number. As of 04/15/2019 there are 59510 of those
#takes book no as an argument, for book 1 the url will be
#https://www.gutenberg.org/ebooks/1
#paras splits by newline
def scrap_book_num(book_num):
       book_num = str(book_num)
       book_url = URL + book_num
       web_request = requests.get(book_url)
       web_html = web_request.text
       try:
           soup = BeautifulSoup(web_html,"html.parser")
       except:
            pass
       links = soup.find_all('a',class_="link")
       out_path = get_path(book_num, output_dir)
       for link in links:
         if "Text" in link.text:
           get_book = link.get('href')
           request = requests.get("https:"+get_book)
           #../is the parent directory, you can change the following line to
           #change output destination
           with open(out_path+"book"+book_num+".txt",'wb') as open_file:
               for chunk in request.iter_content(10000):
                    open_file.write(chunk)
           print("book%s" % book_num + "worked")
           open_file.close()
            
#this gunction calls the text_cleaner.sh script, which converts to the necessary file format for NLTK to 
#to be able to understand the text files, while also removing unecessary repetetive lines relating
#to copyright or project gutenberg.
def clean(begin, end ,output_dir):
    Popen(["bash", "-c", "chmod +x text_cleaner.sh"])
    for i in range(begin,end+1,100):
        clean_path = get_path(i,output_dir)
        call(["bash","./text_cleaner.sh", clean_path])
        print("cleaned", clean_path)

        
"""
This code is multithreading the scrapping to make it run faster, as it is very slow
this might take quite a bit of cpu power
there's a bug on macOS Sierra or higher that has not been fixed as of
04/15/2019, that doesn't allow python to fork sometimes.
running the following line on your command line should fix it:
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
if that doesn't fix it, you can make a regular loop that downloads each book
"""

#this is without multithreading. If the machine is too fast, multithreading can break request limits
# def main():
#     for i in range(begin, end):
#         sleep(0.5)
#         scrap_book_num(i)
#     return

#the reason this is using a while loop not a for loop is that too many
#consecutive http requests raise an error, so it sleeps for a bit and 
#tries the same book again rather than moving onto the next book
def main():
    i = begin
    while i <= end:
        try:
            scrap_book_num(i)
            i+=1
        except:
            sleep(3)
            scrap_book_num(i)

#sometimes one thread finishes late, meaning the next lines are executed
#before all the books are downloaded, which causes an error.
#sleep fixes that.
if __name__=="__main__":
    Popen(["bash", "-c", "chmod +x make_dirs.sh"])
    call(["bash","./make_dirs.sh",output_dir])
    sleep(1)
    main()
    sleep(1)
    clean(begin,end, output_dir)
