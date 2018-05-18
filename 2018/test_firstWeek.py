import unittest
from parser import chunk
from parser import readFile
import csv
class TestCase(unittest.TestCase):
    def setUp(self):
        pass
    def test_output(self):
        sentences = readFile('./verified_analogies.csv')
        base = []
        target = []
        with open('base_target_output.csv') as file:
            readcsv = csv.reader(file, delimiter=',')
            for row in readcsv:
                b = row[2]
                t = row[3]
                base.append(b)
                target.append(t)
        print(len(base), len(target), len(sentences))
        for i  in range(len(sentences)):
            project_base, project_target = chunk(sentences[i])
            print(base[i])
            self.assertEqual(project_base,base[i])
            self.assertEqual(project_target,target[i])
    def test_None(self):
        sentences = readFile('./verified_analogies.csv')
        for s in sentences
            self.assertIsNotNone(chunk(s))

if __name__ == '__main__':
    unittest.main()