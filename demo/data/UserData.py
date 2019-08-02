
import json

USER_DATA_PATH = "../data/user_info_without_moments.json"

# 歌曲数据集
# 提供曲库初始化和歌曲查询的接口
class UserData:
    # 类初始化
    def __init__(self):
        self.user_list = []
        with open(USER_DATA_PATH, encoding='UTF-8') as f:
            line = f.readline()
            while line:
                self.user_list.append(json.loads(line))
                line = f.readline()

    # 按照歌曲ID查找歌曲详细信息
    # @para user_id: 待查询用户的ID
    # @return: 如果查到了则输出对应的用户信息，查不到则输出空词典
    def searchUserById(self, user_id):
        for user in self.user_list:
            if user_id == user["id"]:
                return user
        return {}

if __name__ == "__main__":
    ud = UserData()
    print(ud.searchUserById(922222600))