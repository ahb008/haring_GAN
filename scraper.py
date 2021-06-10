#Python script to scrape The Keith Haring Foundation website to download 200x200 iamges for training set

import requests
from bs4 import BeautifulSoup
from PIL import Image


baseURL = "https://www.haring.com/!/genre/"

genres = ["drawing", "editions", "painting", "public_projects", "sculpture"]

for index, genre in enumerate(genres):

    requestResult = requests.get("https://www.haring.com/!/genre/" + genre)

    soup = BeautifulSoup(requestResult.content, features="html.parser")

    images = soup.findAll("img")
    #Remove first "img" which is not a Haring artwork
    images.pop()

    print(len(images))

    for idx, image in enumerate(images):
        imageURL = image.attrs['src']
        img = Image.open(requests.get(imageURL, stream=True).raw)
        img.save('/Users/andrewbass/Desktop/bad_answers/tech_projects/haring_GAN/data/' + str(index) + str(idx) + ".jpg", 'JPEG')





