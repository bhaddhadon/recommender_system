import requests
# Change the value of experience that you want to test
url = 'http://172.17.0.2:5000/api'

#-------------------------------------------------------------------
# age: numeric
# gender: ['F','M']
# region: int (1 to 23)
#-------------------------------------------------------------------

age = 32
gender = 'F'
region = 3

feature = [age, gender, region]

r = requests.post(url,json={'feature': feature})
print(r.json())