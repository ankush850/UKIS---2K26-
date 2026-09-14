import requests
import json
import time

try:
    print('Calling /api/superresolve...')
    t0 = time.time()
    res = requests.post('http://127.0.0.1:8000/api/superresolve', json={
        'num_mc_samples': 2,
        'scale_factor': 4,
        'model_name': 'hat',
        'apply_unsharp': False,
        'apply_realesrgan_sharpen': False,
        'visualization_mode': 'true_color'
    })
    print('Status:', res.status_code)
    if res.status_code != 200:
        print('Error text:', res.text[:500])
    print('Time taken:', time.time() - t0)
except Exception as e:
    print('Exception:', e)
