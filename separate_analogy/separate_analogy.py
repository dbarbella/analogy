import csv
import statistics


sampleRatings = {}     # dict to hold the data from csv file
incompleteRating = {}
analogySample = {}
notAnalogySample = {}
unsureSample = {}

# Method to load data from csv to sampleRatings
def loadData(fileName):
    with open(fileName, 'r') as file:
        fileReader = csv.reader(file)
        next(fileReader, None)
        duplicates = []
        # reading the csv row
        for row in fileReader:
            id = row[0]
            text = row[1]
            
            if id in duplicates:
                continue
            
            duplicates.append(id)

            # Append if dict has an entry
            if id in sampleRatings:
                if row[2] == '': continue
                try:
                    rating = int(row[2])
                    if rating<1 or rating>3:
                        print("Error: Wrong rating type (Needs to be 1, 2 or 3)\n" + str(row))
                        break
                    sampleRatings[id][1].append(rating)
                
                except ValueError:
                    print("Error: Wrong rating type (Needs to be 1, 2 or 3)\n" + str(row))
            
            # Create a new entry if not found
            else:
                if row[2] == '':
                    continue
                rating = int(row[2])
                sampleRatings[id]=(text,[rating])
                
# filter the data with threshold reviews of (n)
def filterData(threshold):
    # Copy incomplete data to incompleteRating
    for key, value in sampleRatings.items():
        if len(value[1]) < threshold:
            incompleteRating[key] = value
    
    # remove incomplete rating from sampleRatings
    for key in incompleteRating:
        del sampleRatings[key]

# sort analogies by average of ratings 
def analogyRelegate(): 
    for key, value in sampleRatings.items():  
        if statistics.mean(value[1]) < 2:
            analogySample[key] = value
        elif statistics.mean(value[1]) > 2:
            notAnalogySample[key] = value
        elif statistics.mean(value[1]) == 2:
            unsureSample[key] = value 
            
# Method to export dictionary as CSV files        
def exportCSV(fileName, samples):
    with open(fileName, 'w') as csvfile:
        
        writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
        writer.writerow(['id', 'text', 'ratings'])

        for key, value in samples.items():
            text = value[0]
            ratingList = [rating for rating in value[1]]
            outputRow = [key, text]
            outputRow.extend(ratingList)
            writer.writerow(outputRow)
        print(fileName+".csv export sucess.")
    
def main():
    # csv file names to import from
    files = ['Reviewer 1.csv','Reviewer 2.csv','Reviewer 3.csv']

    # loading data from files list
    for fileName in files:
        loadData(fileName)
        
    filterData(3)
    analogyRelegate()
    
    exportCSV('analogy_sample.csv', analogySample)
    exportCSV('not_analogy_sample.csv', notAnalogySample)
    exportCSV('unsure_sample.csv', unsureSample)
    print("Success")
    
main()