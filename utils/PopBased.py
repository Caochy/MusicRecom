import json
import os


path = '/data/disk5/private/xcj/MusicRecom/processdata/DataUser'
outpath = '/data/disk5/private/xcj/MusicRecom/processdata/music_pop.json'

def like(score):
    return True
    if score > 9:
        return True
    else:
        return False

def MusicLikeNum(filelist):
    music_num = {}

    for filename in filelist:
        fin = open(os.path.join(path, filename), 'r')
        for line in fin:
            line = json.loads(line)
            for m in line['music']:
                if like(line['music'][m]):
                    if not m in music_num:
                        music_num[m] = 0
                    music_num[m] += 1
    
    fout = open(outpath, 'w')
    music_num = sorted(music_num.items(), reverse = True, key = lambda x:x[1])
    for m in music_num:
        print('%s\t%d' %(str(m[0]), m[1]), file = fout)


def PopBasedRecom(data, label):
    fin = open(outpath, 'r')
    
    poprank = {}
    for line in fin:
        sid = int(line.split('\t')[0])
        poprank[sid] = len(poprank)

    tmpdata = []
    for i in range(len(data)):
        if int(data[i]) in poprank:
            tmpdata.append((data[i], label[i], poprank[int(data[i])]))
        else:
            print(data[i])

    sorted(tmpdata, key = lambda x:x[2])

    ranksum = 0
    pnum = 0
    for i in range(len(tmpdata)):
        v = tmpdata[i]
        if v[1] == 1:
            ranksum += len(tmpdata) - i - 1
            pnum += 1

    auc = (ranksum - pnum * (pnum + 1)/2) / (pnum * (len(label) - pnum))

    print('auc', auc)



if __name__ == '__main__':
    #MusicLikeNum(['train.json', 'test.json'])
    fin = open('/data/disk5/private/xcj/MusicRecom/processdata/test.txt', 'r')
    data = []
    label = []
    for line in fin:
        line = line.strip().split('\t')
        try:
            tid = int(line[0])
            tl = int(line[1])
            data.append(tid)
            label.append(tl)
        except:
            print(line)
    
    PopBasedRecom(data, label)





