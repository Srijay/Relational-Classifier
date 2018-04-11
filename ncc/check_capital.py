
infile = open("/data/srijayd/local_data/f_r1_r2_th/th/th.csv","r")

for line in infile:
    line = line.split(",")
    mod = line[0]
    head = line[1]
    if(head[0].isupper()):
        print head
