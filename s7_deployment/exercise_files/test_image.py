import requests
response = requests.get('https://imgs.xkcd.com/comics/making_progress.png')


with open(r'img.png','wb') as f:
   f.write(response.content)

