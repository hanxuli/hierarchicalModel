def predataset(file):
    train=[]
    label=[]
    number=[]
    max=0
    with open(file,encoding="utf8") as file:
        for line in file:
            temp=line.split("\t\t")
            train.append(temp[3])
            number.append(len(temp[3]))
            label.append(int(temp[2])-1)
            if max<len(temp[3]):
                max=len(temp[3]);
    print("max",max)
    return  train ,label,number


x_train,y_train,number=predataset("../data/IMDB/test.txt")
print(sum(number)/len(number))