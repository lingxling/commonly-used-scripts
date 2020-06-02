#!/usr/bin/env python3

"""
微信小程序云开发不支持json直接上传，需要改成json.line
"""
import json
import jsonlines

with open('大学排名信息.json', 'r', encoding='utf8')as fp:
    data = json.load(fp)
with jsonlines.open('大学排名信息jl.json', 'w') as writer:
    for info in data:
        writer.write(info)
