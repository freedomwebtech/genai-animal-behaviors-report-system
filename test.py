import cv2
import base64
import os
import time
import threading
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# ✅ Set up Google API Key
os.environ["GOOGLE_API_KEY"] = ""

# ✅ Initialize the Gemini model
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# ✅ Function to send the captured image to Gemini for analysis
def analyze_image_with_gemini(image):
    if image is None:
        return "No image to analyze."

    # Convert the captured image to base64
    _, img_buffer = cv2.imencode('.jpg', image)
    image_data = base64.b64encode(img_buffer).decode('utf-8')

    # Create the message with the image
    message = HumanMessage(
        content=[ 
            {"type": "text", "text": "The agent's task is to detect cows. If there are multiple cows, check for each cow individually and report if they are eating or not. Provide only that information."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}} 
        ]
    )
    
    # Send the message to Gemini and get the response
    response = model.invoke([message])
    return response.content

# ✅ Function to save response to a text file with timestamp
def save_response_to_file(response):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    filename = "gemini_responses.txt"
    
    with open(filename, "a", encoding="utf-8") as file:
        file.write(f"{timestamp} - {response}\n\n")
    
    print(f"Response saved to {filename}")

# ✅ Function to continuously capture images and analyze them
def background_capture(cap):
    while True:
        time.sleep(2)  # Wait for 2 seconds before capturing the next image
        
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
        
        print("Sending the image for analysis...")
        response_content = analyze_image_with_gemini(frame)  # Analyze the image with Gemini
        print("Gemini Response:", response_content)  # Print the response from Gemini
        
        save_response_to_file(response_content)  # Save response to a file

# ✅ Main function to show live feed and start background analysis
def main():
    cap = cv2.VideoCapture('cow.mp4')  # Load video file or replace with 0 for webcam

    if not cap.isOpened():
        print("Error: Unable to access the camera.")
        return

    # Start a background thread to capture and analyze images every 2 seconds
    capture_thread = threading.Thread(target=background_capture, args=(cap,))
    capture_thread.daemon = True  # Ensure the thread exits with the main program
    capture_thread.start()

    # Continuously show the video feed
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.resize(frame, (1020, 500))  # Resize frame for better display
        cv2.imshow("Video Feed", frame)

        # Exit on 'q' key press
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
