import requests
from bs4 import BeautifulSoup
import bs4
'''
爬取中国大学排名
'''


def get_html_test(url):
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        r.encoding = r.apparent_encoding
        return r.text
    except:
        return ""


def fill_univ_list(ulist, html):
    soup = BeautifulSoup(html, "html.parser")
    for tr in soup.find("tbody").children:
        if isinstance(tr, bs4.element.Tag):
            tds = tr("td")
            ulist.append([tds[0].string, tds[1].string, tds[3].string])

def print_univ_list(ulist, num):
    print("{:^10}\t{:^6}\t{:^10}".format("排名", "学校名称", "总分"))
    for i in range(num):
        u = ulist[i]
        print("{:^10}\t{:^6}\t{:^10}".format(u[0], u[1], u[2]))



if __name__ == '__main__':
    uinfo = []
    url = "http://www.zuihaodaxue.cn/zuihaodaxuepaiming2016.html"
    html = get_html_test(url)
    fill_univ_list(uinfo, html)
    print_univ_list(uinfo, 100)