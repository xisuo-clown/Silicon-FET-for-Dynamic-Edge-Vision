import requests
import re
import time


# 获取汇率的函数
def get_exchange_rate():
    url = "https://www.google.com/finance/quote/SGD-CNY?sa=X&ved=2ahUKEwiI9efxz-2LAxVExzgGHRdOM7cQmY0JegQIChAu"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"
    }
    response = requests.get(url, headers=headers)
    text = response.text
    pattern = r'data-last-price="([^"]+)"'
    rate = re.search(pattern, text)

    if rate:
        exchange_rate = float(rate.group(1))
        print("SGD 对 CNY 的汇率: {:.3f}".format(exchange_rate))
        return exchange_rate
    else:
        print("未找到汇率信息")
        return None


# 发送消息到 QQ 机器人的函数
def send_qq_message(message):
    # 这里是你的 QQ 机器人接口地址和群号或好友 QQ 号
    url = "http://localhost:5700/send_private_msg"  # go-cqhttp 接口地址
    data = {
        "user_id": "553596264",  # 发送的目标 QQ 号
        "message": message
    }
    response = requests.post(url, data=data)
    if response.status_code == 200:
        print("成功发送消息到 QQ")
    else:
        print("发送失败")


# 定时提醒的函数
def remind_exchange_rate():
    while True:
        rate = get_exchange_rate()
        if rate:
            message = f"当前 SGD 对 CNY 的汇率是: {rate:.3f}"
            send_qq_message(message)

        time.sleep(1800)  # 每隔 30 分钟 (30 分钟 = 1800 秒)

import requests

bot_token = '8134339595:AAEDL_uU_aSO8UAem9ZHAYjvkaLGhTbqp-k'
chat_id = '5520269358'
rate=get_exchange_rate()
message = "新币对人民币汇率：{:.3f}".format(rate)

url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
params = {
    'chat_id': chat_id,
    'text': message
}

response = requests.get(url, params=params)
print(f"消息发送成功，响应: {response.json()}")

