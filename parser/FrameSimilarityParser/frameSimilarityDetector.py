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
        
    def __hash__(self):
        return hash(self._name)
    
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
        return self._elements[:]
    
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
        
        ownEOSum= len(oElements) + len(oRelations)
        
        # If they share the same frame elements, this weight is taken into accounbt
        for i in self.getElements():
            if i in oElements:
                overLap+=.1
        
        # If they share the same relation, this weight is taken into account
        for i in self.getRelations():
            if i in oRelations:
                overLap += 1
                
        return abs(overLap - ownEOSum)/ownEOSum > .97
    
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
    

def getRelation(start, frames, depth, curDepth=0):
    '''
        Returns a list of frames related to the given start frame, indirectly
        out to a depth
        
        start = a frame from which to generate relations
        frames = the list of frames to find relations in
        depth = the number of direct relations to follow out
    '''
    if curDepth >= depth:
        return []
    relations = start.getRelations()
    for ind, relation in enumerate(relations):
        relation = list(filter(lambda x: x.getName() == relation, frames))
        if not len(relation):
            continue
        relation = relation[0]
        relations[ind] = [relation]+ getRelation(relation, frames, depth, curDepth+1)
    return relations

def getSimilarityAtDepth(start, frames, depth, curDepth=0):
    '''
        getSimilarityAtDepth:        
            Returns a set of frames related or "similar" to the starting frame.
            Frame similarity is defined by frame's internal similarity function
            (if compared frames have a large number of similar frame elements
            and relations)
        
        start = a frame from which to generate relations
        frames = the list of frames to find relations in
        depth = the number of direct relations to follow out
    '''
    if curDepth >= depth:
        return []
    # Remove the start frame from frames left to check
    frames = [frame for frame in frames if frame != start]
    # A list of frames directly similar to the starting frame
    similars = []
    
    # Go through all the frames in frames and add immediately similar frames and their similarities to the list of similars
    
    for frame in frames:
        if start.similarTo(frame):
            similars.append([frame] + list(getSimilarityAtDepth(frame, frames, depth, curDepth+1)))
    
    # Returns 1-d list of similar frames
    return {i for e in similars for i in e}
            
            

'''
    if curDepth >= depth:
        return []
    relations = start.getRelations()
    for ind, relation in enumerate(relations):
        relation = list(filter(lambda x: x.getName() == relation, frames))[0]
        similars = list(filter(lambda x: x.similarTo(relation), frames))
        relations[ind] = [relation]+ [i for b in [getRelation(similar, frames, depth, curDepth+1) for similar in similars] for c in b for i in c ]
    return [i for b in relations for i in b]
'''


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
    
    a = list(getSimilarityAtDepth(frames[1], frames, 2))
    #print(a[:10])
    #print(len(a))
    file.close()
    
    return frames
    
if __name__ == '__main__':
    a = main()