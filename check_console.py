from selenium import webdriver
from selenium.webdriver.chrome.options import Options

options = Options()
options.add_argument('--headless')
driver = webdriver.Chrome(options=options)
driver.get("http://127.0.0.1:8000")
# trigger analysis
driver.execute_script("document.getElementById('dashboard-container').style.display = 'block'; document.getElementById('dashboard-container').style.opacity = '1';")
import time
time.sleep(2)

logs = driver.get_log('browser')
for entry in logs:
    if entry['level'] == 'SEVERE':
        print('ERROR:', entry['message'])

driver.quit()
