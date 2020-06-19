import numpy as np
import cv2
import collections
from collections import deque
import copy
print(cv2.__version__)

background = None
global x1,y1,x2,y2

'''
-------------------------------------------------------
# Function that detects whether a pixel belongs to the skin based on RGB values
PARAMS: 
curr_frame_base - the source color image
roi - colored, smaller region of interest where we'll do skin detection
dst - the destination grayscale image where skin pixels are colored white and the rest are colored black
RETURN: thresholded skin detected frame, current base frame, region of interest box/frame that we
determined to be the hand 
-------------------------------------------------------
'''

def mySkinDetect(curr_frame, curr_frame_base):
    # Surveys of skin color modeling and detection techniques:
    # 1. Vezhnevets, Vladimir, Vassili Sazonov, and Alla Andreeva. "A survey on pixel-based skin color detection techniques." Proc. Graphicon. Vol. 3. 2003.
    # 2. Kakumanu, Praveen, Sokratis Makrogiannis, and Nikolaos Bourbakis. "A survey of skin-color modeling and detection methods." Pattern recognition 40.3 (2007): 1106-1122.
    dst = np.zeros((curr_frame_base.shape[0], curr_frame_base.shape[1], 1), dtype = "uint8")
    for i in range(curr_frame_base.shape[0]):
        for j in range(curr_frame_base.shape[1]):
            #b,g,r = curr_frame_base[i,j]
            b = int(curr_frame_base[i,j][0])
            g = int(curr_frame_base[i,j][1])
            r = int(curr_frame_base[i,j][2])
            # if(r>95 and g>40 and b>20 and max(r,g,b)-min(r,g,b)>15 and abs(r-g)>15 and r>g and r>b):
            if(r>125 and g>70 and b>50 and max(r,g,b)-min(r,g,b)>15 and abs(r-g)>15 and r>g and r>b):
                dst[i,j] = 255

    # threshold hand 
    ret, skin_thresh = cv2.threshold(dst, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # find hand contours to outline hand
    contours_skin, hierarchy = cv2.findContours(skin_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


    # grab 2 largest contours based on area size 
    contours_skin = sorted(contours_skin, key = cv2.contourArea, reverse = True)[:2]


    # loop over our contours
    for idx,c in enumerate(contours_skin):
        # approximate the contour
        peri = cv2.arcLength(c, True)
        # approx = cv2.approxPolyDP(c, 0.015 * peri, True)
        approx = cv2.approxPolyDP(c, 0.019 * peri, True)

        # if our approximated contour has 8 points, then
        # we can assume that we have found circle (head)
        if len(approx) == 8:
            del contours_skin[idx]
            break


    # get and draw bounding box from skin contours ON CURR FRAME 
    try: 
        x,y,w,h = cv2.boundingRect(contours_skin[0])

        cv2.rectangle(curr_frame,(x,y),(x+w,y+h),(0,0,255),2)

        x_down = x
        y_down = y
        x_up = x+w
        y_up = y-h
        r,h,c,w = y_down , (y_up-y_down) , x_down , (x_up-x_down)

        roi = skin_thresh[r:r+h, c:c+w]

        if roi.shape[0] == 0: 
            roi = skin_thresh


    except Exception as e: 
        print('') 
        print(e)
        print('exception, making roi = skin_thresh')
        roi = skin_thresh

    #draw contours on skin mask 
    # cv2.drawContours(curr_frame, contours_skin + [145 + -95], -1, (0, 255, 0),2)
    try:
        # cv2.drawContours(curr_frame, contours_skin + [135 + -75], -1, (255, 0, 0),2)
        cv2.drawContours(curr_frame, contours_skin, -1, (255, 0, 0),2)
        # cv2.drawContours(curr_frame, hand_contour, -1, (255, 0, 0),2)
    except: 
        print('')


    return dst,curr_frame_base, roi


'''
-------------------------------------------------------
# Function that does frame differencing between the current frame and the previous frame
# prev - the previous color image
# curr - the current color image
# dst - the destination grayscale image where pixels are colored white if the corresponding pixel intensities in the current
# and previous image are not the same
-------------------------------------------------------
'''
def myFrameDifferencing(prev, curr):
    # For more information on operation with arrays: 
    # http://docs.opencv.org/modules/core/doc/operations_on_arrays.html
    dst = cv2.absdiff(prev, curr)
    dst = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    # _, dst = cv2.threshold(dst, 50, 255, cv2.THRESH_BINARY)
    _, dst = cv2.threshold(dst, 20, 255, cv2.THRESH_BINARY)
    return dst


'''
-------------------------------------------------------
# Function that accumulates the frame differences for a certain number of pairs of frames
PARAMS: 
mh - vector of frame difference images
dst - the destination grayscale image to store the accumulation of the frame difference images
RETURNS: summed motion energy frame, ratio of white to black pixels to determine static or not
-------------------------------------------------------
'''
def myMotionEnergy(mh):
    # the window of time is 3
    mh0 = mh[0]
    mh1 = mh[1]
    mh2 = mh[2]
    dst = np.zeros((mh0.shape[0], mh0.shape[1], 1), dtype = "uint8")
    for i in range(mh0.shape[0]):
        for j in range(mh0.shape[1]):
            if mh0[i,j] == 255 or mh1[i,j] == 255 or mh2[i,j] == 255:
                dst[i,j] = 255

    # total sum of dst matrix if all black (no motion)
    all_black = 150*150*255

    # ratio of white (movement) to black (no movement)
    ratio = dst.sum()/all_black

    # threshold hand movements and find area of contour 


    return dst, ratio

'''
--------------------------------------------------------------
Function that returns string of detected/most accurate gesture 
PARAMS: 
- frame: big current frame
- roi: smaller reigion of interest 
- frameDest: thresholded roi from frame
- mySkin: skin thresholded roi 
- templates_dict: copy of original template dicts to store accuracies
- template_objs: original dictionary of all templates
- type: 'static' or 'dynamic'
RETURNS: string of gesture name, float accuracy 
--------------------------------------------------------------
'''
def match_gesture(frame, frameDest, mySkin, templates_dict, template_objs, type):
    #inital gesture is Null 
    gesture = ''
    threshold = 0.8
    
    if type == 'static':
        # make a copy of template objs so we don't modify original 
        templates_dict = copy.deepcopy(templates_dict)

        # get frame shape 
        frame_width, frame_height = frame.shape[0], frame.shape[1]

        # get roi size 
        skin_height, skin_width = mySkin.shape[0], mySkin.shape[1]


        for name, t in templates_dict.items():
            template_list = t[0]

            # compute template match for every template
            for template in template_list: 
                acc = 0

                # resize template to match roi size 
                template = cv2.resize(template, (skin_width, skin_height))


                # match template between skin threshold and current template 
                # returns matrix res which is accuracy at each point in frame
                res = cv2.matchTemplate(mySkin,template,cv2.TM_CCOEFF_NORMED)

                # get location where accuracy is highest, which in sq. norm is min location
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                top_left = min_loc
                bottom_right = (top_left[0] - frame_height, top_left[1] - frame_width)

                #get accuracy 
                try: 
                    # acc = res[min_loc[0]][min_loc[1]]
                    acc = res[max_loc[0]][max_loc[1]]
                    # if acc == 1.0: 
                    #     acc = 0
                except: 
                    acc = 0
                
                # append accuracy to list of accuracies for that template 
                templates_dict[name][1].append(acc)

        # sum overall accuracies 
        for name, t in templates_dict.items():
            # print('t: ', t[1])
            overall_accuracy = sum(t[1])/len(t[1])
            templates_dict[name] = overall_accuracy



        # template_obj now looks like: 
        # {'a': 0.6514961570501328, 'b': 0.5898155570030212, 'c': 0.7233994007110596, 'd': 0.07028644531965256, 'love': 1.0}
        # sort by highest accuracy to lowest 
        templates_dict = sorted(templates_dict.items(), key = lambda x: x[1], reverse = True)
        print(templates_dict)

        # return gesture with highest accuracy and accuracy percentage
        highest_accuracy = templates_dict[0]
        gesture = highest_accuracy[0]
        
        return gesture, highest_accuracy[1]

    # if a dynamic gesture 
    else: 

        # get contours in thresholded roi 
        contours_dynamic, hierarchy = cv2.findContours(mySkin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # sort contours, get max which should be your hand 
        contours_dynamic = sorted(contours_dynamic, key = cv2.contourArea, reverse = True)[0]

        # get area of your hand contour 
        area = cv2.contourArea(contours_dynamic)


        # find extreme points of contour 
        leftmost = tuple(contours_dynamic[contours_dynamic[:,:,0].argmin()][0])
        rightmost = tuple(contours_dynamic[contours_dynamic[:,:,0].argmax()][0])
        topmost = tuple(contours_dynamic[contours_dynamic[:,:,1].argmin()][0])
        bottommost = tuple(contours_dynamic[contours_dynamic[:,:,1].argmax()][0])

        width = np.abs(rightmost[0]-leftmost[0])
        height = np.abs(topmost[1] - bottommost[1])


        # check for hello gesture where width will be greater than 200 
        if height < 280 and width >= 200: 
            gesture = 'hello'
            highest_accuracy = 'N/A'
        # # check for the gesture 'No' (moving fist up and down) where height will be greater 
        # elif height > 250 and width <150: 
        #     gesture = 'No'
        #     highest_accuracy = 'N/A'

        # hand is not doing any of those gestures
        else: 
            gesture = 'calcuating...'
            highest_accuracy = 'N/A'

        return gesture, highest_accuracy

        


'''
-----------------------------------------------------------------------------------
main frame - shows gesture, type and accuracy text, along with bounding boxes and contours
base frame - frame that purely shows just the orig frame with no extra stuff
-----------------------------------------------------------------------------------
'''
def main():

    # load all templates 
    template_objs = {'a':([],[]), 'b':([],[]), 'c':([],[]), 'd':([],[])}
    dynamic_templates = {'hello': ([], [])}

    all_templates = {'a':['gestures/a0.png', 'gestures/a1.png', 'gestures/a2.png', 'gestures/a3.png', 'gestures/a4.png', 'gestures/a5.png', 'gestures/a6.png', 'gestures/a7.png', 'gestures/a8.png'], 
    'b':['gestures/b0.png', 'gestures/b1.png','gestures/b3.png', 'gestures/b4.png', 'gestures/b5.png'], 
    'c':['gestures/c0.png','gestures/c1.png', 'gestures/c2.png', 'gestures/c3.png', 'gestures/c4.png', 'gestures/c5.png', 'gestures/c6.png', 'gestures/c7.png'],
    'd':['d0.png', 'd1.png', 'd2.png', 'd3.png', 'd4.png', 'd5.png', 'd6.png', 'd7.png', 'd8.png', 'd9.png'],
    'hello': ['gestures/hello0.png', 'gestures/hello1.png', 'gestures/hello2.png', 'gestures/hello3.png', 'gestures/hello4.png', 'gestures/hello5.png', 'gestures/hello6.png', 'gestures/hello7.png', 'gestures/hello8.png', 'gestures/hello9.png']}

    # read in black empty template to determine static or dynamic 
    no_movement = cv2.imread('gestures/no_movement.png', 0)

    for name, temps in all_templates.items(): 
        count = 0
        for t in temps: 

            template_name = cv2.imread(t,0)

            
            if name in template_objs:
                template_objs[name][0].append(template_name)

            if name in dynamic_templates: 
                dynamic_templates[name][0].append(template_name)

            count += 1

    # a) Reading a stream of images from a webcamera, and displaying the video
    # open the video camera no. 0
    # for more information on reading and writing video: http://docs.opencv.org/modules/highgui/doc/reading_and_writing_images_and_video.html
    cap = cv2.VideoCapture(0)
    
    #if not successful, exit program
    if not cap.isOpened():
        print("Cannot open the video cam")
        return -1

    # read a new frame from video
    success, prev_frame = cap.read()

    prev_frame_base = prev_frame.copy()
    
    #if not successful, exit program
    if not success:
        print("Cannot read a frame from video stream")
        return -1
    cv2.namedWindow("frame", cv2.WINDOW_AUTOSIZE)
    
    prev_frame = cv2.resize(prev_frame,(480,300))
    prev_frame_base = cv2.resize(prev_frame_base,(480,300))

    # fMH1 = np.zeros((prev_frame_roi.shape[0], prev_frame_roi.shape[1], 1), dtype = "uint8")
    fMH1 = np.zeros((prev_frame.shape[0], prev_frame.shape[1], 1), dtype = "uint8")
    fMH2 = fMH1.copy()
    fMH3 = fMH1.copy()
    myMotionHistory = deque([fMH1, fMH2, fMH3]) 

    count = 0
    

    while(True):
        #read a new frame from video
        success, curr_frame = cap.read()

        # make a copy of current frame without detailing to set as base frame
        curr_frame_base = curr_frame.copy()

        curr_frame = cv2.resize(curr_frame,(480,300))
        curr_frame_base = cv2.resize(curr_frame_base,(480,300))
        if not success:
            print("Cannot read a frame from video stream")
            break
        
        # flip the frames so that it is not the mirror view
        curr_frame = cv2.flip(curr_frame, 1)
        curr_frame_base = cv2.flip(curr_frame_base, 1)

        # Background differencing on base frame
        # get region of interest from frame differencing 
        frameDest = myFrameDifferencing(prev_frame_base, curr_frame_base)

        #Skin color detection, draw contours on curr_frame mySkinDetect(curr_frame, source)
        mySkin, curr_frame_base, roi = mySkinDetect(curr_frame, curr_frame_base)
        # cv2.imshow('mySkinDetect',mySkin)


        # get motion energy templates, and black and white ratio 
        # of motion energy templates to determine if gesture is moving or not 
        myMotionHistory.popleft()
        myMotionHistory.append(frameDest)
        myMH, ratio = myMotionEnergy(myMotionHistory)
        # cv2.imshow('motion energy: ', myMH)


        # make copy of original templates & their empty accuracies 
        template_objs_copy = {k:v for k,v in template_objs.items()}
        dynamic_templates_copy = {k:v for k,v in dynamic_templates.items()}


        # check motion energy frames ratio 
        # if the ratio, which is just the ratio of white pixels to black pixels 
        # is over a certain threshold, we can assume there is movement in the frame
        if ratio >= 0.4: 
            cv2.putText(curr_frame, 'dynamic', (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

            # template match for dynamic gesture 
            # dynamic_gesture, dynamic_accuracy = match_gesture(curr_frame,myMH, roi, dynamic_templates_copy, template_objs, 'dynamic')
            dynamic_gesture, dynamic_accuracy = match_gesture(curr_frame,roi, myMH, dynamic_templates_copy, template_objs, 'dynamic')

            cv2.putText(curr_frame, dynamic_gesture, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
            cv2.putText(curr_frame, 'accuracy: ' + str(dynamic_accuracy), (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

        
        # detected low movement, so template match for static gesture 
        else: 
            cv2.putText(curr_frame, 'static', (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

            # gesture, static_accuracy = match_gesture(curr_frame, roi, frameDest, mySkin, template_objs_copy, template_objs)
            gesture, static_accuracy = match_gesture(curr_frame, frameDest, roi, template_objs_copy, template_objs, 'static')
            # dynamic_gesture, dynamic_accuracy = match_gesture(curr_frame,myMH, roi, dynamic_templates_copy, template_objs)

            if static_accuracy >= 0.2: 
                # show gesture name
                cv2.putText(curr_frame, gesture, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
            else: 
                # show that it's still calcuating 
                cv2.putText(curr_frame, 'calculating...', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

            if static_accuracy < 0: 
                static_accuracy = -(static_accuracy)
                cv2.putText(curr_frame, 'accuracy: ' + str(static_accuracy), (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
            else: 
                # show gesture name and accuracy 
                cv2.putText(curr_frame, 'accuracy: ' + str(static_accuracy), (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

            
        cv2.imshow('frame', curr_frame)
        prev_frame = curr_frame
        prev_frame_base = curr_frame_base


        #observe keypress 
        k = cv2.waitKey(1) & 0xff
        
        # wait for 'q' key press. If 'q' key is pressed, break loop
        if k == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # print(template_objs)
    main()