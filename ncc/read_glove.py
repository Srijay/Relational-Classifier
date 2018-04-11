import numpy as np
def readStanfordGlove():
    vocabArray = []
    vocabMap = {}
    embeds = []
    infile = open("/data/srijayd/local_data/glove_stanford/GLOVE_6B/glove.6B.300d.txt","r")
    i = 0
    for line in infile:
        line = line.split(" ")
        word = line[0]
        embed = line[1:]
        embed = map(float,embed)
        vocabArray.append(word)
        vocabMap[i] = word
        i+=1
        embeds.append(embed)
        if(i==2):
            break
    embeds = np.array(embeds)
    print vocabArray
    print vocabMap
    print embeds
    

readStanfordGlove()
        
    
