from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import sys
import time

def run_tests():
    print("============================= test session starts ==============================")
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-gpu')
    
    try:
        driver = webdriver.Chrome(options=options)
    except Exception as e:
        print(f"FAILED to initialize Chrome driver: {e}")
        return
        
    driver.implicitly_wait(10)
    
    passed = 0
    total = 3
    start_time = time.time()
    
    try:
        # Test 1: App Loads
        print("test_app_loads ... ", end="", flush=True)
        driver.get("http://localhost:8501")
        wait = WebDriverWait(driver, 15)
        header = wait.until(EC.presence_of_element_located((By.CLASS_NAME, "main-header")))
        assert "Multimodal Interview Analysis System" in header.text
        print("PASSED")
        passed += 1
        
        # Test 2: Video Upload Widget
        print("test_video_upload_present ... ", end="", flush=True)
        driver.get("http://localhost:8501")
        file_input = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='file']")))
        assert file_input is not None
        print("PASSED")
        passed += 1
        
        # Test 3: Analyze Button
        print("test_analyze_button_present ... ", end="", flush=True)
        driver.get("http://localhost:8501")
        wait.until(lambda d: "Analyze Video" in d.page_source)
        print("PASSED")
        passed += 1
        
    except Exception as e:
        print(f"FAILED: {e}")
        
    finally:
        driver.quit()
        
    end_time = time.time()
    duration = end_time - start_time
    print(f"============================== {passed} passed in {duration:.2f}s ==============================")
    
if __name__ == "__main__":
    run_tests()
