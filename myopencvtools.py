import cv2

# ################## VIDEO FUNCTIONS ###############

def initvideostream(source, goalFPS = None):

    stream = cv2.VideoCapture(source)
    # Check if camera opened successfully
    if (stream.isOpened()== False):
        print('Error opening video stream or file')

    if source == 0: #Camara

        if goalFPS is None:

            if stream.get(cv2.CAP_PROP_FPS) != 0:
                goalFPS = stream.get(cv2.CAP_PROP_FPS)
                print(f'Camara FPS is {goalFPS} fps')

            else: print(f'Warning: Unable to get camera FPS')

        else:

            print(f'Trying to set camera fps to goal of {goalFPS}')
            stream.set(cv2.CAP_PROP_FPS, goalFPS)
            if stream.get(cv2.CAP_PROP_FPS) == goalFPS:
                print('DONE')
            else:
                goalFPS = stream.get(cv2.CAP_PROP_FPS)
                print(f'FPS set to camera available rate of {goalFPS} fps')

    # Video
    elif isinstance(source,str) and goalFPS is None:
        
        if stream.get(cv2.CAP_PROP_FPS) != 0:
            goalFPS = stream.get(cv2.CAP_PROP_FPS)
            print(f'FPS is set to the video original rate of {goalFPS} fps')
        else:
            goalFPS = 24
            print('Video did not provide the FPS information')
            print(f'set to default :{goalFPS} fps')
            print('Otherwise please choose as desire fps as argument')

    imgWidth = int(stream.get(cv2.CAP_PROP_FRAME_WIDTH))
    imgHeight = int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT)) 

    print(f'goal FPS: {goalFPS}')
    print(f'Video Source Dimensions: {imgWidth} x {imgHeight}')
    print('press q to quit video stream')
    
    return stream, goalFPS, imgWidth, imgHeight

def initvideowriter(file = 'output.mp4', FPS = 15, dimensions = (640, 480), isColor = True, verbose = False):
    
    # Define the codec and create VideoWriter object
    # 3IVD: FFmpeg DivX (MS MPEG-4 v3), FMP4: FFMpeg, FFV1: FFMPEG Codec. 
    # .avi, .mpg, .mp4
    fourcc = cv2.VideoWriter_fourcc(*'FMP4')
    out = cv2.VideoWriter(file,fourcc, FPS, dimensions, isColor)
    if verbose: print(f'Video writer parameters: file: {file}, FPS: {FPS}, dimensions: {dimensions}, isColor: {isColor}') 
    return out

def getframe(stream, goalFPS = None, looptime = None, skippframes = True, verbose = False):

    #If looptime and goalFPS availables:

    if looptime is not None and goalFPS is not None:
        
        # Too slow
        if goalFPS*looptime > 1:

            # skipp  frames if wished
            if skippframes:

                framesToSkipp = round(goalFPS*looptime)

                if verbose: print(f'skipping every {framesToSkipp} frames to reach {goalFPS} fps refered to video')

                for n in range(framesToSkipp):

                    grabbed = stream.grab()
         
        #Loop too fast, wait
        elif goalFPS*looptime <= 1:

            wait = 1/goalFPS-looptime
            
            if verbose: print(f'Loop to fast, waiting {wait*1000} ms to reach {goalFPS} fps')

            time.sleep(wait)
        
    grabbed, frame = stream.read()

    return grabbed, frame
