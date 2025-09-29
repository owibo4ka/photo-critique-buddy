import requests
import os

def test_upload():
    # URL of your API
    url = "http://localhost:8001/analyze"
    
    # Path to a test image (replace with actual path)
    # You can use any image file on your computer
    image_path = "/path/to/your/test/image.jpg"
    
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"Image file not found: {image_path}")
        print("Please update the image_path variable with a real image file path")
        return
    
    try:
        # Open and send the file
        with open(image_path, "rb") as image_file:
            files = {"file": image_file}
            response = requests.post(url, files=files)
        
        # Print the response
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the API. Make sure your FastAPI server is running on port 8001")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_upload()
