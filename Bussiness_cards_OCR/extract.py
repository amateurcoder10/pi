#Tesseract works best on images which have a DPI of at least 300 dpi, so it may be beneficial to resize images.

from PIL import Image
import sys
import pytesseract
#import pyttsx
from gtts import gTTS
import pygame

if len(sys.argv) != 2:
    print ("%s input_file output_file" % (sys.argv[0]))
    sys.exit()
else:
    text=pytesseract.image_to_string(Image.open(sys.argv[1]))
    print (text)
    #print(text.type)
    tts = gTTS(text, lang='en')
    tts.save("output.mp3")
  
    pygame.mixer.init()
    pygame.mixer.music.load("output.mp3")
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy() == True:
        continue

    '''
    engine = pyttsx.init()
    engine.say(text)
    engine.runAndWait()
    '''
