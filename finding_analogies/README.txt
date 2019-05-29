README

gutenberg_scrapper.py

The code requires Python3, and several packages as specified in the requirements.

The code is ran like:

python gutenberg_scrapper.py output_dir begin_num end_num

Where output_dir is where you want the text files to be scraped.
If you want them to run in the directory the script is ran,
you should set output_dir to ./ .

begin_num is the lowest book id you want to scrape. If you want to scrap all,
or just start from the beginning anyway, this number should be 1.

end_num is the highest book id you want to scrape. If you want to scrape all
(59510 books, 5 GB) you can type [all]. Otherwise, write any number less than that.
The code will scrape all books with ids between the begin_num and end_num.

The program will create its own hierarchy of sub-folders to accommodate easier
search, and make it easier for the computer to load the GUI (this is an actual concern).

Some gutenberg links do not have texts but video or audio. These will be skipped,
and files with that book_id will not exist. This is important to remember
when writing code using scraped texts, since accessing a non-existing
file can raise an error.

scrapeWrodHunter.py

This is wordHunt.py adapted for the output files and directory hierarchy of gutenberg_scrapper.py
It will give both .txt and a csv of analogies, with what file and where in the file
the analogies are in.
The output will be stored in the indicated output directory.
The program will create subdirectories for the texts
(if the subdirectories don't already exist)
It's ran as:
python scrapeWordHunt.py input_dir output_dir begin_num end_num
where:
input_dir: directory where the scraped texts are.
  The program can get to the proper subdirectory its self.
ouput_dir: directory where the analogy .txt and .csv files will be
  stored. Will have the same sub-directory structure as gutenberg_scrapper.py
begin: lowest book id
end: highest book id
The code will extract likely analogies from all existing books
with bookids between begin and end.

punctuation_fix.py

TAKES A SINGLE CSV FILE AS AN ARGUMENT, NOT A DIRECTORY!
Many Gutenberg texts have extra spaces between punctuations signs
due to the quality of the scan, which can make the sentences hard
to read.  This script removes the extra spaces. It is not very fast,
so should only be ran when necessary, such as texts that will
be labeled by a human reader.
It is run as:
python punctuation_fix.py in_path (out_path)
where:
in_path: path to the csv file. Takes a single csv file, not a directory.
out_path: This one is optional. Output path to the resulting csv file.
  Takes a single csv file, not a directory. If a path is not specified,
  the new csv will have the same path as the old csv (basically in place)
