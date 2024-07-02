""" import os
import pygame
import time

pygame.mixer.init()
is_within_threshold = False
pygame.mixer.music.load("alert_sound.wav")
pygame.mixer.music.play(-1)  
print('done')
""" """ def update_proximity_flag(offset_x, offset_y, threshold_distance):
    global is_within_threshold
    is_within_threshold = abs(offset_x) <= threshold_distance and abs(offset_y) <= threshold_distance

def control_audio_playback(is_within_threshold):
    if is_within_threshold:
        pygame.mixer.music.play()  # Play the audio
        print('Audio Playing')
    else:
        pygame.mixer.music.stop() 
        print('Audio Stopped')


#control_audio_playback(True) """ """ """

import pygame
import time

# Initialize pygame
pygame.mixer.init()

# Load the WAV file
audio_file_path = "alert_sound.wav"  # Update with your audio file path
pygame.mixer.music.load(audio_file_path)

# Play the audio
pygame.mixer.music.play()

# Flag to track whether the audio is paused
is_paused = False

# Loop to continuously check for user input
while True:
    user_input = input("Press 'p' to pause/resume or 'q' to quit: ")
    if user_input.lower() == 'p':
        if is_paused:
            pygame.mixer.music.unpause()  # Resume playback
            is_paused = False
            print("Audio Resumed")
        else:
            pygame.mixer.music.pause()  # Pause playback
            is_paused = True
            print("Audio Paused")
    elif user_input.lower() == 'q':
        break  # Exit the loop and quit
    else:
        print("Invalid input. Press 'p' to pause/resume or 'q' to quit.")

# Clean up resources
pygame.mixer.quit()

