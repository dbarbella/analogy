#changes the directory to the output directory
cd $1
#converts to utf-8 by removing the first 3 bytes
for f in *.txt; do
    tail -c +4 "$f" > "${f%.txt}.nobom"
done

#removes .txt files
find . -name "*.txt" -maxdepth 1 -type f -print0 | xargs -0 /bin/rm -f

#changes the .nobom files to .txt files
for f in *.nobom; do
    mv -- "$f" "${f%.nobom}.txt"
done


for f in *_no_bom.txt; do
    mv -- "$f" "${f%_no_bom.txt}.txt"
done

#removes gutenber/copy right related lines from the files
#I think this removes some files entirely
#fixin that would require changing this lines I think
#to run this in mac, add ['.bak'] between -i and the regular expression.
#then uncomment the comment after this loop
for i in book*; do
  if grep -q "START OF THIS PROJECT GUTENBERG\|END OF THIS PROJECT GUTENBERG" $i
  then
    sed -i '1,/START OF THIS PROJECT GUTENBERG/d;/END OF THIS PROJECT GUTENBERG/, $d;/^$/d; /End of the Project/, $d' $i
  fi
done

# find . -name "*bak" -type f -print0 | xargs -0 /bin/rm -f
