import json

SONG_DATA_PATH = "../data/song_info.json"

# 歌曲数据集
# 提供曲库初始化和歌曲查询的接口
class SongData:
    # 类初始化
    def __init__(self):
        self.song_list = []
        with open(SONG_DATA_PATH, encoding='UTF-8') as f:
            line = f.readline()
            while line:
                self.song_list.append(json.loads(line))
                line = f.readline()

    # 按照歌曲ID查找歌曲详细信息
    # @para music_id: 待查询歌曲的ID
    # @return: 如果查到了则输出对应的歌曲词典信息，查不到则输出空词典
    def searchMusicById(self, music_id):
        for music in self.song_list:
            if music_id == music["k_song_id"]:
                return music
        return {}
    
    def searchMusicBySinger(self, singer):
        result = []
        for music in self.song_list:
            if singer == music["singer_title"]:
                result.append(music)
        return result

if __name__ == "__main__":
    sd = SongData()
    print(sd.searchMusicById(3170504))
