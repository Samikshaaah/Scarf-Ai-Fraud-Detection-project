import requests, io
csv = io.StringIO('feature1,feature2,label\n1,2,0\n3,4,1')
files = {'file': ('test.csv', csv.getvalue())}
resp = requests.post('http://127.0.0.1:8000/train/', files=files)
print(resp.status_code)
print(resp.text[:200])
