# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 08:46:01 2019

@author: josep
"""

from ast import literal_eval

class Frame:
    def __init__(self, name=None, relations=None, elements=None):
        '''
        Creates a frame instance using given relations and elements if given
        '''
        self._name = name
        self._relations = relations if relations else []
        self._elements = elements if elements else []
        
    def __repr__(self):
        return self._name + '|' + str(self._relations) + '|' + str(self._elements)
        
    def addElements(self, elements):
        # Adds list of element to frame
        self._elements += elements[:]
        
    def addElement(self, element):
        # Adds element to frame
        self._elements += [element]
        
    def addRelations(self, relations):
        # Adds list of relations to frame
        self._relations += relations[:]
    
    def addRelation(self, relation):
        # Adds relation to frame
        self._relations += [relation]
    
    def setName(self, name):
        # Sets name of frame
        self._name = name
        
    def getElements(self):
        # Returns list of elements in frame
        return self.elements[:]
    
    def getName(self):
        # Returns name of frame
        return self._name
        
    def getRelations(self):
        # Return relations of frame
        return self._relations[:]
    
    def similarTo(self, other):
        '''
            Judges whether a frame is similar to another frame based on number of
            keyword similarities in the frames
        '''
        overLap = 0
        oElements = other.getElements()
        oRelations = other.getRelations()
        
        # If they share the same frame elements, this weight is taken into accounbt
        for i in self.getElements():
            if i in oElements:
                overLap+=1
        
        # If they share the same relation, this weight is taken into account
        for i in self.getRelations():
            if i in oRelations:
                overLap += 5
                
        return overLap > len(self._Relations)*3/5 + len(self.addElements)*.6
    
    def __eq__(self, other):
        return self.getName() == other.getName()
    
class FrRelationParser:
    def __init__(self, file):
        self._file = iter(file)
        self._curLine = next(self._file).split(' ')
        self._nextLine = next(self._file)
    
    def has_next(self):
        return self._nextLine != '</frameRelations>'
    
    def proceed(self):
        assert self.has_next(), "Error: No remaining commands"
        self._curLine = self._nextLine.strip().split(' ')
        self._nextLine = next(self._file)

    def i_type(self):
        if len(self._curLine) > 1:
            if 'frameRelation' in self._curLine[0]:
                return 'C_Fr_Relation'
            elif 'FERelation' in self._curLine[0]:
                return 'C_FE_Relation'
        else:
            return 'C_Other'
        
    def getSubId(self):
        return self._curLine[1][self._curLine[1].index('"')+1:-1]
    
    def getSupId(self):
        return self._curLine[2][self._curLine[1].index('"')+1:-1]
    
    def getSubName(self):
        return self._curLine[3][self._curLine[3].index('"')+1:-1]
    
    def getSuperName(self):
        return self._curLine[4][self._curLine[4].index('"')+1:-1]
    
    def getID(self):
        return self._curLine[5][self._curLine[5].index('"')+1:-1]
    
def loadFrames(fileName):
    '''
    Takes in frame output text file in frameName [Frame Relation List] [Frame Element List]
    format and returns a list of frames filled with file's specified contents
    '''
    
    frames = []
    
    with open(fileName, 'r') as inFile:
        for line in inFile.readlines():
            line = line.split('|')
            if len(line) > 1:
                frames.append(Frame(name=line[0], relations = literal_eval(line[1]), elements=literal_eval(line[2].strip())))
            
    return frames
    
def main():
    '''
    Parses frRelation.xml into list of frames and writes the frames to a file
    '''
    fileName = 'frRelation.xml'
    file = open(fileName, 'r')
    parser = FrRelationParser(file.readlines())
    parser.proceed()
    parser.proceed()
    frames = []
    
    while parser.has_next():
        if parser.i_type() == 'C_Fr_Relation':
            frames.append(Frame(name = parser.getSubName(), relations = [parser.getSuperName()]))
        elif parser.i_type() == 'C_FE_Relation':
            frames[-1].addElement(parser.getSubName())
        parser.proceed()
        
    with open('frRelationOut.txt', 'w') as outFile:
        for frame in frames[1:]:
            outFile.write(str(frame)+'\n')
            outFile.write('-'*20+'\n')

    file.close()
    
if __name__ == '__main__':
    main()