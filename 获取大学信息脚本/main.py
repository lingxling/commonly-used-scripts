#!/usr/bin/env python3

"""
参考链接:
    https://blog.csdn.net/weixin_44024393/article/details/92143429
    https://www.jianshu.com/p/62104d636fa4
"""

import requests
import json
import csv


# 构建请求头
headers = {
    'Accept': '*/*',
    'Accept-Encoding': 'gzip, deflate',
    'Accept-Language': 'zh-CN,zh;q=0.9',
    'Connection': 'keep-alive',
    'contentType': 'application/x-www-form-urlencoded; charset=utf-8',
    'Cookie': 'cfm-major=true',
    'Host': 'gaokao.afanti100.com',
    'media': 'PC',
    'Referer': 'http://gaokao.afanti100.com/university.html',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.157 Safari/537.36',
    'X-Requested-With': 'XMLHttpRequest',
}


# 声明一个列表存储字典
data_list = []


def get_index():

    page = 1
    while True:
        if page > 188:
            break
        url = 'http://gaokao.afanti100.com/api/v1/universities/?degree_level=0&directed_by=0' \
              '&university_type=0&location_province=0&speciality=0&page={}'.format(page)
        # page自增一实现翻页
        page += 1
        # 请求url并返回的是json格式
        resp = requests.get(url, headers=headers).json()

        # 取出大学所在的键值对
        university_lsts = resp.get('data').get('university_lst')
        if university_lsts:
            get_info(university_lsts)
        else:
            continue


def get_info(university_lsts):
    # 判断列表是否不为空
    if university_lsts:
        # 遍历列表取出每个大学的信息
        for university_lst in university_lsts:
            # 声明一个字典存储数据
            data_dict = {'name': university_lst.get('name'),
                         'ranking': university_lst.get('ranking'),
                         'tag_lst': university_lst.get('tag_lst'),
                         'key_major_count': university_lst.get('key_major_count'),
                         'graduate_program_count': university_lst.get('graduate_program_count'),
                         'doctoral_program_count': university_lst.get('doctoral_program_count'),
                         'is_211': university_lst.get('is_211'),
                         'is_985': university_lst.get('is_985'),
                         'location_province': university_lst.get('location_province'),
                         'location_city': university_lst.get('location_city'),
                         'university_type': university_lst.get('university_type')}
            if data_dict['location_province'] in ['北京', '天津', '上海', '重庆']:
                data_dict['location_province'] += '市'
            else:
                data_dict['location_province'] += '省'

            logo = requests.get(university_lst.get('logo_url')).content
            with open('images/' + data_dict['name'] + '.jpg', 'wb') as f:
                f.write(logo)
            # 根据大学名称搜索经纬度
            latlng = get_latlng(data_dict['name'])
            if latlng is not None:  # 如果搜不到地址，直接丢弃。
                data_dict['latitude'] = latlng[0]
                data_dict['longitude'] = latlng[1]
                data_list.append(data_dict)


def get_latlng(address):
    """
    返回经纬度信息
    """
    ak = 'VNfGC2LBq6uYTO6sDKqQoA1WfiZiIFiv'
    url = 'http://api.map.baidu.com/geocoding/v3/?address={inputAddress}&output=json&ak={myAk}'.format(inputAddress=address,myAk=ak)
    res = requests.get(url)
    json_data = json.loads(res.text)

    if json_data['status'] == 0:
        lat = json_data['result']['location']['lat']  # 纬度
        lng = json_data['result']['location']['lng']  # 经度
    else:
        return None
    return lat, lng


def save_file():
    # 将数据存储为json文件
    with open('大学排名信息.json', 'w', encoding='utf-8') as f:
        json.dump(data_list, f, ensure_ascii=False, indent=4)
    print('json文件保存成功')

    # 将数据存储为csv文件
    # 表头
    title = data_list[0].keys()
    with open('大学排名信息.csv', 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, title)
        # 写入表头
        writer.writeheader()
        # 写入数据
        writer.writerows(data_list)
    print('csv文件保存成功')


def main():
    get_index()
    save_file()


if __name__ == '__main__':
    main()

