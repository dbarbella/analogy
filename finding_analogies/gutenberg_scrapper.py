import requests
from bs4 import BeautifulSoup
from multiprocessing import Pool

"""
This is to scrap the all the English books on gutenberg.org as of 04/15/2019
"""

NUM_BOOKS = 59510
URL = "https://www.gutenberg.org/ebooks/"
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
       soup = BeautifulSoup(web_html,"html.parser")
       links = soup.find_all('a',class_="link")
       for link in links:
         if "Text" in link.text:
           get_book=link.get('href')
           request = requests.get("https:"+get_book)
           #../is the parent directory, you can change the following line to
           #change output destination
           with open("../../../bookScraping/book"+book_num+".txt",'wb') as open_file:
               for chunk in request.iter_content(10000):
                    open_file.write(chunk)
           open_file.close()




#I am multithreading the scrapping to make it run faster, as it is very slow
#this might take quite a bit of cpu power
#there's a bug on macOS Sierra or higher that has not been fixed as of
#04/15/2019, that doesn't allow python to fork sometimes.
#running the following line on your command line should fix it:
#export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
#if that doesn't fix it, you can make a regular loop that downloads each book
def main():
    for i in range(38800,NUM_BOOKS,5):
      with Pool(5) as p:
          action = p.map(scrap_book_num, [x for x in range(i,i+5)])




if __name__=="__main__":
    main()