README

gutenberg_scrapper.py

The code requires Python3, and several packages as specified in the requirements.

The code is ran like:
python gutenberg_scrapper.py output_dir begin_num end_num

Where output_dir is where you want the text files to be scrapped.
If you want them to run in the directory the script is ran,
you should set output_dir to ./ .

begin_num s the lowest book id you want to scrape. If you want to scrap all,
or just start from the beginning anyway, this number should be 1.

end_num is the highest book id you want to scrape. If you want to scrape all
(59510 books, 25 GB) you can type [all]. Otherwise, write any number less than that.
The code will scrape all books with ids between the begin_num and end_num.

The program will create its own hierarchy of sub-folders to accommodate easier
search, and make it easier for the computer to load the GUI (this is a serious concern).

Some gutenberg links do not have texts but video or audio. These will be skipped,
and files with that book_id will not exist. This is important to remember
when writing code using scraped texts, since accessing a non-existing
file can raise an error.

IF YOU ARE RUNNING ON MACOS, AND YOU GET AN ERROR, RUN THIS IN YOUR COMMAND LINE:
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES

If for any reason the code stops running, you can restart with begin as the last book downloaded. 
It is possible that the code will stop running after the downloads finished, but before the cleaning finished.
In this case, call clean.py manually.

clean.py
Python script to call clean.sh. Is ran as:

python clean.py diectory_to_be_cleaned low_book_num high_book_num

It will remove BOM (which is necessary for NLTK to be able to read the files) and remove
extra lines that project gutenberg adds to the files.

clean.sh will not run on some systems, most notably MacOS, without some tweaks. 
You can tweak it by following the comments in clean.sh.
It should run on GNU.

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
  The program can get to the proper subdirectory itself.
ouput_dir: directory where the analogy .txt and .csv files will be
  stored. Will have the same sub-directory structure as gutenberg_scrapper.py
begin: lowest book id
end: highest book id
The code will extract likely analogies from all existing books
with bookids between begin and end.

pick_random.py

pick random analogies form all scraped analogies and writes them into a csv.
Called like:
python pick_random.py input_dir output_dir num_per_csv num_total_csv

where:
input_dir: directory with the scraped analogies from which the random analogies will be picked from
output_dir: directory where the randomly picked analogies will be stored
num_per_csv: how many analogies per csv file
num_total_csv: how many total csv files.


punctuation_fix.py

TAKES A SINGLE CSV FILE AS AN ARGUMENT, NOT A DIRECTORY!
Many Gutenberg texts have extra spaces between punctuation signs
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
  
context.py

Is used in pick_random, gets a number of previous and following sentences for each csv. This is not a script users are supposed to call, so the number of sentences is hard-coded. To change them, change NUM_PREV and NUM_AFTER in the python file, at the top.


