import cv2
import youtube_dl
import numpy as np
import easyocr
import time

start_time = time.time()

def download_twitch_vod(url, output_path="."):
    ydl_opts = {
        'outtmpl': f'{output_path}/output.mp4',
        'format': 'best',  # Download best quality available
    }
    
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        try:
            ydl.download([url])
            print("Download completed!")
        except Exception as e:
            print(f"Error: {e}")

def preprocess_for_player_names(frame):

    height, width = frame.shape[:2]
    
    # Define regions for player names (adjust these coordinates based on your layout)
    left_region = frame[0:200, 0:600]  # Top-left area
    right_region = frame[0:200, width-600:width]  # Top-right area
    
    processed_regions = []
    
    for region in [left_region, right_region]:
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(region)
        
        # Morphological operations to clean up (idk what this does tbh)
        kernel = np.ones((2,2), np.uint8)
        cleaned = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel)
        
        # Resize for better OCR (make text larger)
        scaled = cv2.resize(cleaned, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
        
        processed_regions.append(scaled)
    
    return processed_regions

def sort_text_by_height(image_path, reader):
    
    results = reader.readtext(image_path, detail=1,
                              # Performance optimizations
                              width_ths=0.7,  # Adjust text width threshold
                              height_ths=0.7,  # Adjust text height threshold
                              paragraph=False,  # Disable paragraph grouping
                              batch_size=1)    # Process one image at a time)
    
    text_with_height = []
    for (bbox, text, confidence) in results:
        bbox = np.array(bbox)
        height = max(bbox[:, 1]) - min(bbox[:, 1])
        
        text_with_height.append({
            'text': text,
            'confidence': confidence,
            'height': height,
            'bbox': bbox
        })
    
    # Sort by height (tallest first)
    sorted_by_height = sorted(text_with_height, key=lambda x: x['height'], reverse=True)
    
    return sorted_by_height

def find_keyframes_by_histogram(frame, prev_frame, threshold=0.5):
    
# Calculate histogram
    hist = cv2.calcHist([frame], [0, 1, 2], None, [50, 50, 50], [0, 256, 0, 256, 0, 256])
    prev_hist = cv2.calcHist([prev_frame], [0, 1, 2], None, [50, 50, 50], [0, 256, 0, 256, 0, 256])

    # Compare histograms using correlation
    correlation = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
    
    # If correlation is low, frames are different (potential keyframe)
    is_keyframe = (correlation < threshold)

    return is_keyframe, correlation

def frames_to_minutes(frames, fps):
    minutes, seconds = divmod(int(frames / fps), 60)
    return minutes, seconds

vod_url = "https://www.twitch.tv/videos/2538239636"
download_twitch_vod(vod_url, ".")

cap = cv2.VideoCapture('output.mp4')
if not cap.isOpened():
        print("Error: Could not open video")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("Video is at", fps, "fps", width, "x", height)

total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
print("Video duration is", total_frames / fps / 60, "minutes")

reader = easyocr.Reader(['en'], gpu=True, verbose=False)

current_frame = 0
key_frame_list = []
prev_frame = np.zeros((height, width, 3), dtype=np.uint8) # I think this creates an empty frame
matches_list = []
timestamp_list = []

while (current_frame < total_frames):
    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
    ret, frame = cap.read()
    is_keyframe = find_keyframes_by_histogram(frame, prev_frame)

    if (is_keyframe):
        key_frame_list.append(current_frame)
    
    current_frame += fps * 60 * 2 # Increment by 2 minutes

keyframe_cnt = 1
for key_frame in key_frame_list:
    print("Processing", keyframe_cnt, "of", len(key_frame_list), "keyframes")
    cap.set(cv2.CAP_PROP_POS_FRAMES, key_frame)
    ret, frame = cap.read()
    areas_of_interest = preprocess_for_player_names(frame)

    text = sort_text_by_height(areas_of_interest[0], reader)
    if (text): # if text was found in a frame
        name1 = text[0]["text"]

    text = sort_text_by_height(areas_of_interest[1], reader)
    if (text): # if text was found in a frame
        name2 = text[0]["text"]

    timestamp = str(frames_to_minutes(key_frame, fps)[0]) + ":" + str(frames_to_minutes(key_frame, fps)[1]).zfill(2)
    matchup = name1 + " vs " + name2
    if (matchup not in matches_list):
        matches_list.append(matchup)
        timestamp_list.append(timestamp)

    keyframe_cnt += 1

for i in range(len(matches_list)):
    print(matches_list[i] + " " + timestamp_list[i])

cap.release()
del reader

end_time = time.time()
execution_time = end_time - start_time
print(f"Program ran for {execution_time:.2f} seconds")