
import jetson.inference
import jetson.utils

net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)
img_path = "/home/nvidia/jetson-inference/examples/wfq_detection/brown_bear.jpg"
img = jetson.utils.loadImage(img_path)

detections = net.Detect(img)

for detection in detections:
	class_label  = net.GetClassDesc(detection.ClassID)
	confidence = detection.Confidence
	left = detection.Left
	top = detection.Top
	right = detection.Right
	bottom = detection.Bottom
	width = right - left
	height = bottom - top
	area = width * height
	center = ((right + left)/2, (bottom + top)/2)


print(f"--ClassID: {class_label}")
print(f"--Confidence={confidence:.5f}")
print(f"--Left={left:.5f}")
print(f"--Top={top:.5f}")
print(f"--Right={right:.5f}")
print(f"--Bottom={bottom:.5f}")
print(f"--Width={width:.5f}")
print(f"--Height={height:.5f}")
print(f"--Area={area:.5f}")
print(f"--Center={center:.5f}")
