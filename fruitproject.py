import jetson.inference
import jetson.utils as jetu
import numpy as np
import cv2

# Declare the detector
net = jetson.inference.detectNet(argv=['--model=fruits.onnx', '--labels=labels.txt', '--input-blob=input_0', '--output-cvg=scores', '--output-bbox=boxes'])

# Declare the camera and window
camera = jetson.utils.videoSource("/dev/video0")
display = jetson.utils.videoOutput()
window = jetson.utils.videoOutput()

# Apple Oxidation
brown_lower = np.array([10, 150, 50], np.uint8)
brown_upper = np.array([30, 200, 205], np.uint8)

# Orange Fungus
white_lower = np.array([0, 0, 200], np.uint8)
white_upper = np.array([179, 5, 255], np.uint8)

# Banana Spots
black_lower = np.array([0, 0, 0], np.uint8)
black_upper = np.array([50, 150, 50], np.uint8)

while True:
    # Capture frames
    img = camera.Capture()

    # Keys
    keym = 0
    keyn = 0
    keyb = 0

    # Convert images to numpy arrays
    frame = jetu.cudaToNumpy(img)

    # Perform detection
    detections = net.Detect(img, overlay='none')

    # Declare lists
    xlist = []
    ylist = []

    # If there are detections
    if detections:
        for detection in detections:
            # Determine the fruit type
            class_id = detection.ClassID

            # If it's an apple
            if class_id == 1:
                # Extract coordinates
                xim, yim = detection.Left, detection.Top
                xfm, yfm = detection.Width + xim, detection.Top + detection.Height

                # Save coordinates to avoid errors
                xlist.append(xim)
                xlist.append(xfm)
                ylist.append(yim)
                ylist.append(yfm)

                xminm, xmaxm = int(min(xlist)), int(max(xlist))
                yminm, ymaxm = int(min(ylist)), int(max(ylist))

                # Extract region of interest
                roim = frame[yminm:ymaxm, xminm:xmaxm]

                # Convert to HSV
                hsvm = cv2.cvtColor(roim, cv2.COLOR_RGB2HSV)

                # Threshold
                maskm = cv2.inRange(hsvm, brown_lower, brown_upper)

                # Contours
                contoursm, _ = cv2.findContours(maskm, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

                # Sort
                contoursm = sorted(contoursm, key=lambda x: cv2.contourArea(x), reverse=True)

                for contm in contoursm:
                    # Extract area
                    aream = cv2.contourArea(contm)

                    if 50 <= aream <= 5000:
                        # Detect bad areas
                        xsim, ysim, anchom, altom = cv2.boundingRect(contm)

                        # Display errors
                        jetu.cudaDrawRect(img, (xim + xsim, yim + ysim, xim + xsim + anchom, yim + ysim + altom), (255, 0, 0, 80))

                        # Bad Apple
                        print("BAD APPLE")

                        keym = 1

                        # Convert CUDA image
```python
                        ven = jetu.cudaFromNumpy(roim)
                        # Display region of interest
                        window.Render(ven)

                if keym == 0:
                    # Good Apple
                    jetu.cudaDrawRect(img, (xim, yim, xfm, yfm), (0, 255, 0, 80))
                    print("GOOD APPLE")

            # If it's a banana
            elif class_id == 2:
                # Extract coordinates
                xib, yib = detection.Left, detection.Top
                xfb, yfb = detection.Width + xib, detection.Top + detection.Height

                # Save coordinates to avoid errors
                xlist.append(xib)
                xlist.append(xfb)
                ylist.append(yib)
                ylist.append(yfb)

                xminb, xmaxb = int(min(xlist)), int(max(xlist))
                yminb, ymaxb = int(min(ylist)), int(max(ylist))

                # Extract region of interest
                roib = frame[yminb:ymaxb, xminb:xmaxb]

                # Convert to HSV
                hsvb = cv2.cvtColor(roib, cv2.COLOR_RGB2HSV)

                # Threshold
                maskb = cv2.inRange(hsvb, black_lower, black_upper)

                # Contours
                contoursb, _ = cv2.findContours(maskb, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

                # Sort
                contoursb = sorted(contoursb, key=lambda x: cv2.contourArea(x), reverse=True)

                for contb in contoursb:
                    # Extract area
                    areab = cv2.contourArea(contb)

                    if 10 <= areab <= 50000:
                        # Detect bad areas
                        xsib, ysib, anchob, altob = cv2.boundingRect(contb)

                        # Display errors
                        jetu.cudaDrawRect(img, (xib + xsib, yib + ysib, xib + xsib + anchob, yib + ysib + altob), (255, 0, 0, 80))

                        # Bad Banana
                        print("BAD BANANA")

                        keyb = 1

                        # Convert CUDA image
                        ven = jetu.cudaFromNumpy(roib)
                        # Display region of interest
                        window.Render(ven)

                if keyb == 0:
                    # Good Banana
                    jetu.cudaDrawRect(img, (xib, yib, xfb, yfb), (0, 255, 0, 80))
                    print("GOOD BANANA")

            elif class_id == 3:
                # Extract coordinates
                xin, yin = detection.Left, detection.Top
                xfn, yfn = detection.Width + xin, detection.Top + detection.Height

                # Save coordinates to avoid errors
                xlist.append(xin)
                xlist.append(xfn)
                ylist.append(yin)
                ylist.append(yfn)

                xminn, xmaxn = int(min(xlist)), int(max(xlist))
                yminn, ymaxn = int(min(ylist)), int(max(ylist))

                # Extract region of interest
                roin = frame[yminn:ymaxn, xminn:xmaxn]

                # Convert to HSV
                hsvn = cv2.cvtColor(roin, cv2.COLOR_RGB2HSV)

                # Threshold
                maskn = cv2.inRange(hsvn, white_lower, white_upper)

                # Contours
                contoursn, _ = cv2.findContours(maskn, cv2.RETR_TREE, cv2.CH```python
                # Sort
                contoursn = sorted(contoursn, key=lambda x: cv2.contourArea(x), reverse=True)

                for contn in contoursn:
                    # Extract area
                    arean = cv2.contourArea(contn)

                    if 10 <= arean <= 5000:
                        # Detect bad areas
                        xsin, ysin, anchon, alton = cv2.boundingRect(contn)

                        # Display errors
                        jetu.cudaDrawRect(img, (xin + xsin, yin + ysin, xin + xsin + anchon, yin + ysin + alton), (255, 0, 0, 80))

                        # Bad Orange
                        print("BAD ORANGE")

                        keyn = 1

                        # Convert CUDA image
                        #ven = jetu.cudaFromNumpy(roin)
                        # Display region of interest
                        #window.Render(ven)

                if keyn == 0:
                    # Good Orange
                    jetu.cudaDrawRect(img, (xin, yin, xfn, yfn), (0, 0, 255, 80))
                    print("GOOD ORANGE")

            # Render the image
            display.Render(img)

            # Update the window
            window.SetStatus("Bad Fruit Detection")

            # Process any keyboard inputs
            if display.IsStreaming():
                if display.IsStreamingComplete() or (keyb == 1 and keym == 1 and keyn == 1):
                    break
