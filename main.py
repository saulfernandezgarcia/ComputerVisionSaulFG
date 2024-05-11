import os
import shutil
import cv2
import numpy as np
from matplotlib import pyplot as plt
import tkinter as tk
from tkinter import filedialog
import tkinter.messagebox
import threading

output_folder = ""

# Saúl Fernández García - 20611765
# Github: saulfernandezgarcia

#---------------------------------------------------------------------------------------------

def extractFrames(file_path, everySeconds):
    '''
    Related links:
    https://docs.opencv.org/4.x/d0/da7/videoio_overview.html
    https://docs.opencv.org/4.x/d4/da8/group__imgcodecs.html
    https://www.geeksforgeeks.org/python-opencv-cv2-imwrite-method/
    '''

    directory_path = os.path.dirname(file_path)
    # Get the filename without extension
    filename = os.path.splitext(os.path.basename(file_path))[0]

    # Create a folder with the same name as the video in the directory path
    output_folder = os.path.join(directory_path, f"{filename}_output")
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    # Create folder for the specific output:
    frame_output_folder = os.path.join(output_folder, f"{filename}_frame_output")
    os.makedirs(frame_output_folder, exist_ok=True)

    # Try to open the video file
    videoCapture = cv2.VideoCapture(file_path)
    if not videoCapture.isOpened():
        print("Error: Unable to open video file. Check if path is correct")
        return
    else:
        print("Found")

    # fps = number of frames per second
    fps = videoCapture.get(cv2.CAP_PROP_FPS)

    # Calculate frame interval: want to get x frame every "everySeconds"
    frameInterval = int(fps) * everySeconds
    frameCount = 0
    frameSavedId = 0

    # Obtain the frames
    while True:
        status_label.config(text=f"Processing frames...", fg="blue")
        status_label.update()

        successReading, frame = videoCapture.read()

        if not successReading:
            break
        else:
            # Extract frames at desired interval:
            if frameCount % frameInterval == 0:
                # Save the frame
                specificFrame = os.path.join(frame_output_folder, "frame_{a}.jpg".format(a = frameSavedId))
                cv2.imwrite(specificFrame, frame)
                print("Frame {a} saved.".format(a = frameSavedId))

                frameSavedId += 1
            frameCount += 1

    # After video is over, free the video capture
    print("- Done obtaining frames from " + filename)
    videoCapture.release()

    return filename, output_folder, frame_output_folder

def imagesFromFolder(frameFolder, images):
    imageFileList = os.listdir(frameFolder)
    for image in imageFileList:
        aux = cv2.imread(frameFolder + "/" + image)
        images.append(aux)

def matcherSelection():
    selection = matcherChoice.get()
    print("Chosen matcher ")
    if selection == 1:
        print("FB")
        return "FB"
    elif selection == 2:
        print("BF")
        return "BF"

def siftHomographyBlending(left, right, matcher, minimumMatches, id):
    status_label.config(text=f"Performing SIFT...", fg="blue")
    status_label.update()

    # Initiate SIFT detector
    # https://github.com/opencv/opencv/issues/16736
    sift = cv2.SIFT_create()

    # Find the keypoints and descriptors with SIFT. Necessary for looking for matches.
    keyPoints1, descriptor1 = sift.detectAndCompute(left, None)
    keyPoints2, descriptor2 = sift.detectAndCompute(right, None)

    matches = 0
    if matcher == "BF":
        '''
        Brute forcing matching: https://docs.opencv.org/3.4/dc/dc3/tutorial_py_matcher.html
        '''

        bf = cv2.BFMatcher()
        nNeighToReturn = 2
        matches = bf.knnMatch(descriptor1, descriptor2, k=nNeighToReturn)
    elif matcher == "FB":
        '''
        Flann-based: 
        https://docs.opencv.org/4.x/d5/d6f/tutorial_feature_flann_matcher.html
        https://www.geeksforgeeks.org/python-opencv-flannbasedmatcher-function/
        An estimator for nearest neighbours without brute-forcing
        '''

        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)

        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(descriptor1, descriptor2, k=2)

    # Store all the good matches as per Lowe's ratio test.
    goodMatches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            goodMatches.append(m)


    if(len(goodMatches) > minimumMatches):
        status_label.config(text=f"Finding homography...", fg="blue")
        status_label.update()

        # Extract point matches from the previously found keypoints.
        srcPointMatches = np.float32([keyPoints1[m.queryIdx].pt for m in goodMatches]).reshape(-1, 1, 2)
        dstPointMatches = np.float32([keyPoints2[m.trainIdx].pt for m in goodMatches]).reshape(-1, 1, 2)

        # Homography + RANSAC
        homography1, mask1 = cv2.findHomography(srcPointMatches, dstPointMatches, cv2.RANSAC, 5.0)

        # For match plotting (inspiration from lab sessions)
        matchesMask = mask1.ravel().tolist()
        draw_params = dict(matchColor=(0, 255, 0),      # draw matches in green color
                           singlePointColor=None,
                           matchesMask=matchesMask,     # draw only inliers
                           flags=2)
        matchedPoints = cv2.drawMatches(left, keyPoints1, right, keyPoints2, goodMatches, None, **draw_params)
        plt.figure()
        plt.imshow(matchedPoints, "gray")
        plt.xticks([]), plt.yticks([])
        print("Matches are done")

        '''
        Preparations for second homography. The left image will be the cumulative result of
        stitching pictures, thus the choice for keeping its width and height.
        The offset will allow room for the stitched images, and their values impact on the image's final dimensions.
        '''
        print("Left shape: " + str(left.shape))
        print("Right shape: " + str(right.shape))
        w = left.shape[1]
        h = left.shape[0]
        offsetX = w // 3 * 2
        offsetY = h // 3

        # Homography and RANSAC
        '''
        Use the previously found destination points (make a copy) but add them an offset
        to create space for the stitched image. It also finds homography for these new points
        and the source ones and warps the perspective accordingly to stitch the final image.
        '''
        dstPointMatches2 = dstPointMatches.copy()
        dstPointMatches2[:, :, 0] = dstPointMatches2[:, :, 0] + offsetX
        dstPointMatches2[:, :, 1] = dstPointMatches2[:, :, 1] + offsetY
        ransacThreshold = 5.0
        homography2, mask2 = cv2.findHomography(srcPointMatches, dstPointMatches2, cv2.RANSAC, ransacReprojThreshold=ransacThreshold)
        imgFinal = cv2.warpPerspective(left, homography2, (w + offsetX, h + offsetY))

        plt.figure()
        plt.imshow(imgFinal)
        plt.xticks([]), plt.yticks([])
        plt.show()

        '''
        Perform feathering for blending
        https://stackoverflow.com/questions/55066764/how-to-blur-feather-the-edges-of-an-object-in-an-image-using-opencv
        '''

        '''
        Prepare feathering:
        The featherMask will be used to control the blending between left and right in imgFinal.
        '''
        blendRadius = 5
        area = 2
        featherMask = np.zeros((imgFinal.shape[0], imgFinal.shape[1]), dtype=np.float32)
        featherWidth = min(blendRadius, imgFinal.shape[1] // area)
        # From left to right prepare a linear gradient in the mask
        featherMask[:, :featherWidth] = np.linspace(1, 0, featherWidth)
        # From right to left prepare a linear gradient in the mask
        featherMask[:, -featherWidth:] = np.linspace(0, 1, featherWidth)

        '''
        Apply feathering to the region of the imgFinal where the overlapping takes place.
        '''
        # Adapt data types
        imgFinal = imgFinal.astype(np.float32)
        # Apply feathering to each color channel separately
        imgFinal[:, :, 0] *= featherMask
        imgFinal[:, :, 1] *= featherMask
        imgFinal[:, :, 2] *= featherMask

        '''
        Resize the right image to match the dimensions of imgFinal.
        This causes some information loss, but it is a necessary step for generating a regular image.
        '''
        right_resized = cv2.resize(right, (imgFinal.shape[1], imgFinal.shape[0]))

        # Blend images by averaging pixel values in the overlap region while adding pixels of each color channel
        imgFinal[:, :, 0] += right_resized[:, :, 0]
        imgFinal[:, :, 1] += right_resized[:, :, 1]
        imgFinal[:, :, 2] += right_resized[:, :, 2]

        '''
        Normalize pixel values in imgFinal:
        Because the operations might yield pixel values that cannot be represented, it is necessary
        to reduce them to an available domain.
        '''
        imgFinal = np.clip(imgFinal, 0, 255).astype(np.uint8)

        print("Done stitching")

        print("imgFinal shape: " + str(imgFinal.shape))

        plt.figure()
        plt.imshow(imgFinal, cmap='gray')
        plt.xticks([]), plt.yticks([])
        plt.show()

        return imgFinal

def frameStitching(images, filePath, outputFolder, chosenMatcher, minimumMatches):
    # https://www.opencvhelp.org/tutorials/advanced/image-stitching/

    directory_path = os.path.dirname(filePath)
    # Get the filename without extension
    filename = os.path.splitext(os.path.basename(filePath))[0]

    # Create folder for the specific output:
    panorama_output_folder = os.path.join(outputFolder, f"{filename}_panorama_output")
    os.makedirs(panorama_output_folder, exist_ok=True)

    counter = 0
    lastMerged = images[1]
    print(len(images))
    for image in images[1:]:
        print("Stitch: " + str(counter))

        lastMerged = siftHomographyBlending(lastMerged, image, chosenMatcher, minimumMatches, counter)
        counter += 1

        status_label.config(text=f"Stitching frame " + str(counter), fg="blue")
        status_label.update()


    print("------------------")
    panorama_picture_output_path = os.path.join(panorama_output_folder, chosenMatcher + "_panorama.jpg")
    cv2.imwrite(panorama_picture_output_path, lastMerged)
    print("Panorama image saved to " + panorama_output_folder)

    status_label.config(text=f"Process completed", fg="green")
    status_label.update()

def process_video_task(file_path):
    global output_folder
    everySeconds = 2
    fileName, output_folder, frameOutputFolder = extractFrames(file_path, everySeconds)

    images = []
    imagesFromFolder(frameOutputFolder, images)

    minimumMatches = 10

    chosenMatcher = matcherSelection()

    frameStitching(images, file_path, output_folder, chosenMatcher, minimumMatches)

    # Display completion message
    tkinter.messagebox.showinfo("Panoram-ing complete!", "Your panorama is ready!")

    # Reenable the "Select Video File" button
    select_button.config(state=tk.NORMAL)
    # Reenable the "Open Output Folder" button
    open_folder_button.config(state=tk.NORMAL)
    # Reenable the bf_matcher radio button and fb_matcher radio button. https://wiki.tcl-lang.org/page/tkinter.Radiobutton
    bf_matcher.config(state=tk.NORMAL)
    # Reenable the fb_matcher radio button
    fb_matcher.config(state=tk.NORMAL)

def process_video():
    global output_folder
    file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.mov;*.avi")])
    file_path = file_path.replace("/", "\\")
    print(file_path)

    if file_path:

        # Disable the "Select Video File" button
        select_button.config(state=tk.DISABLED)
        # Disable the "Open Output Folder" button
        select_button.config(state=tk.DISABLED)
        # Disable the bf_matcher radio button
        bf_matcher.config(state=tk.DISABLED)
        # Disable the fb_matcher radio button
        fb_matcher.config(state=tk.DISABLED)

        status_label.config(text="Processing video...", fg="blue")

        argsToPass = [file_path]
        processing_thread = threading.Thread(target=process_video_task, args = argsToPass)
        processing_thread.start()
    else:
        print("No chosen path / Path could not be fetched")

def open_output_folder():
    # Open the output folder using the file explorer
    global output_folder
    os.startfile(output_folder)

#---------------------------------------------------------------------------------------------

# Thank you to the Tinker Documentation: https://docs.python.org/3/library/tkinter.html

# Create the GUI
root = tk.Tk()
root.title("Video-To-Panorama - Saul FG")

# Welcome label
welcome_label = tk.Label(root,
                         text="Welcome to Saúl Fernández García's Video-To-Panorama compact program!",
                         font=("Lucida Calligraphy", 18),
                         wraplength=500)
welcome_label.pack(pady=8, padx=10)

instructions_title = "Instructions"
instructions_title_label = tk.Label(root,
                                    text=instructions_title,
                                    font=("Lucida Calligraphy", 15),
                                    justify="left",
                                    wraplength=500)
instructions_title_label.pack(pady=(0, 0), padx=10, anchor="w")

# Instructions
instructions_body_text = """1. Click the "Select Video File" button to choose a video file.
2. Once selected, your video will be processed immediately!
3. The processing will yield both the chosen video frames and the panoramas of the video (with Brute-Force matching and Flann-Based matching). These will be saved in a folder under the same name as the video.
4. You may click on the "Open Output Folder" button to see the results.
Advice: please remember to make copies of the outputs to your desired location in your system.
Important: PLEASE keep the video between 5 and 10 seconds long.
"""
instructions_body_label = tk.Label(root,
                                   text=instructions_body_text,
                                   font=("Arial", 15),
                                   justify="left",
                                   wraplength=700)
instructions_body_label.pack(pady=(0, 0), padx=10, anchor="w")

# Create radio buttons
matcherChoice = tk.IntVar()
fb_matcher = tk.Radiobutton(root, text="Flann-Based Matcher", font=("Arial", 9), variable=matcherChoice, value=1)
fb_matcher.pack(anchor=tk.W)
bf_matcher = tk.Radiobutton(root, text="Brute Force Matcher", font=("Arial", 9), variable=matcherChoice, value=2)
bf_matcher.pack(anchor=tk.W)
matcherChoice.set(1)

# Status label
status_label = tk.Label(root, text="", font=("Arial", 12))
status_label.pack(pady=5)

# Video file selection button
select_button = tk.Button(root,
                          text="Select Video File",
                          font=("Lucida Calligraphy", 13),
                          command=process_video,
                          state=tk.NORMAL)
select_button.pack(pady=10)

# Button to open the output folder; it will not appear until the processing is done.
open_folder_button = tk.Button(root,
                               text="Open Output Folder",
                               font=("Lucida Calligraphy", 13),
                               command=open_output_folder,
                               state=tk.DISABLED)
open_folder_button.pack(pady=10)

# Separator
separator_text = """fe"""
separator_label = tk.Label(root,
                           text=separator_text,
                           font=("Wingdings 2", 15))
separator_label.pack(pady=(0, 0), padx=350, anchor="w")

# Acknowledgements
acknowledgements_text = """By Saúl Fernández García.
20611765 scysf2@nottingham.edu.cn s.fernandezg.2021@alumnos.urjc.es
Coursework for: Computer Vision COMP3065 2023-2024
"""
acknowledgements_label = tk.Label(root,
                                  text=acknowledgements_text,
                                  font=("Arial", 10),
                                  justify="left")
acknowledgements_label.pack(pady=(0, 0), padx=10, anchor="w")

#---------------------------------------------------------------------------------------------

root.update_idletasks()
windowWidth = root.winfo_reqwidth()
windowHeigth = root.winfo_reqheight()
root.geometry(str(windowWidth)+"x"+str(windowHeigth))

root.mainloop()

#---------------------------------------------------------------------------------------------
