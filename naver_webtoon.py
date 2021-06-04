import os
import requests
import cv2
import numpy as np
from PIL import Image
from bs4 import BeautifulSoup
from itertools import count

def NAVER_webtoon_downloader(title_id):
    # costomized Header
    headers = {'Referer': 'http://comic.naver.com/index.nhn'}

    # background color for png file
    WHITE = (255, 255, 255)

    # URL for NAVER webtoon
    url = 'http://comic.naver.com/webtoon/detail.nhn'

    # for checking last episode
    ep_list = []

    for no in count(1):
        params = {'titleId': title_id, 'no': no}
        html = requests.get(url, params=params).text
        soup = BeautifulSoup(html, 'html.parser')

        # webtoon title & episode title
        wt_title = soup.select('.detail h2')[0].text
        ep_title = soup.select('.tit_area h3')[0].text
        wt_title = wt_title.split()[0]
        ep_title = ' '.join(ep_title.split()).replace('?', '')

        # check if this episode is last or not
        if ep_title in ep_list:
            break

        ep_list.append(ep_title)

        # episode images
        img_path_list = []

        img_list = [tag['src'] for tag in soup.select('.wt_viewer > img')]
        for ind,img in enumerate(img_list):

            # save the images
            img_name = os.path.basename(img)
            #img_path = os.path.join(wt_title, img_name)

            #dir_path = os.path.dirname(img_path)
            file_path = os.path.join("webtoon_crop",str(no-1))
            if not os.path.exists(file_path):
                os.makedirs(file_path) 
            #img_path_list.append(img_path)

            # if os.path.exists(img_path):
            #     continue

            
            img_data = requests.get(img, headers=headers).content
            res=cv2.imdecode(np.frombuffer(img_data,dtype=np.uint8),cv2.IMREAD_COLOR)
            cv2.imwrite(os.path.join(file_path,str(ind)+".png"),res)

            # with open(img_path, 'wb') as f:
            #     f.write(img_data)

        # im_list = []
        # for img_path in img_path_list:
        #     im = Image.open(img_path)
        #     im_list.append(im)

        # make canvas for appending images
        # canvas_size = (
        #     max(im.width for im in im_list),
        #     sum(im.height for im in im_list)
        # )
        # canvas = Image.new('RGB', canvas_size)
        # top = 0

        # save the webtoon
        # for im in im_list:
        #     canvas.paste(im, (0, top))
        #     top += im.height
        # canvas.save(dir_path + '\/' + ep_title + '.png')

        # delete all temporally images for webtoon
        # for img_path in img_path_list:
        #     os.remove(img_path)

        print(wt_title + ' ' + ep_title + ' is downloaded.')

    print('All episode is downloaded completely.')

NAVER_webtoon_downloader(703846)