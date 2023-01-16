import requests
response = requests.get(
   'https://api.github.com/search/repositories',
   params={'q': 'requests+language:python'},
)
print(response.status_code)

if response.status_code == 200:
       print('Success!')
elif response.status_code == 404:
   print('Not Found.')

response = response.json()
print(response)