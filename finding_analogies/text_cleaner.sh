#this code is for GNU, and running it on other UNIX based systems may require some tewaking. It will run on tools/hopper.

#changes the directory to the output directory
cd $1


#this regular expression is a match for the first three bytes of a BOM text file.
#By removing those (if they exist), the file is converted to a utf-8 file.
#the reason this is not done in place is convenience and enabling it to run on different operating systems
#the additional cost of this is very small
for f in ./*.txt; do
    sed '1s/^\xEF\xBB\xBF//' < "$f" > "${f%.txt}.nobom"
done


#removes the .txt files
find . -name "*.txt" -maxdepth 1 -type f -print0 | xargs -0 /bin/rm -f

#changes the .nobom files to .txt files
for f in ./*.nobom; do
    mv -- "$f" "${f%.nobom}.txt"
done


 
#removes gutenber/copy right related lines from the files
#to run this in macOS, add ['.bak'] between -i and the regular expressions in the then statements.
#then uncomment the comment starting with find . -name '*bak" after this loop

for i in ./book*; do
  if grep -q "START OF THIS PROJECT GUTENBERG|END OF THIS PROJECT GUTENBERG" $i
  then
    sed -i '1,/START OF THIS PROJECT GUTENBERG/d;/END OF THIS PROJECT GUTENBERG/, $d;/^$/d; /End of the Project/, $d' $i
  fi
done



# find . -name "*bak" -type f -print0 | xargs -0 /bin/rm -f
