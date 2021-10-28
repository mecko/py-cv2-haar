import cv2
import os

samples = "samples/"

for f in os.listdir(samples):
    # skip dot files.
    if f.startswith("."):
        continue

    # read image file.
    img = cv2.imread(samples+f)

    # convert to grayscale for haas.
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # load haar classifier.
    tracker = cv2.CascadeClassifier("haar/face.xml")
    # fine all elements in the grayscale image that match the classifier.
    elements = tracker.detectMultiScale(gray_img)

    # watermark the output with the total count.
    cv2.putText(img, "Total %d %s" % (len(elements), "faces"), (100, 100), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

    # add a rectangle and a tag for every element found.
    for (x, y, w, h) in elements:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 255), 1)
        cv2.putText(img, "face", (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 0.7, (255, 0, 255), 1)

    # display output.
    cv2.imshow('>_<', img)
    cv2.waitKey()
