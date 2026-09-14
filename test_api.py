import requests
import json
try:
    print('Calling /api/superresolve...')
    res = requests.post('http://127.0.0.1:8000/api/superresolve', json={})
    print(res.status_code)
    print(res.text)
except Exception as e:
    print('Exception:', e)
