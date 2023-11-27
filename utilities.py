import numpy as np
import matplotlib.pyplot as plt
import urllib.request
import json
import cv2
import tensorflow as tf
from sklearn.cluster import MeanShift, estimate_bandwidth
import json

def draw_level(filename, root_peg_width, outline_color=(0,0,0)):
  # load json file
  data = json.load(open(filename))

  # data has 3 keys: 'filename', 'pegs', 'palette'
  # fetch image from url filename
  image_url = data['filename']

  # fetch image from url
  urllib.request.urlretrieve(image_url, 'image.jpg')

  # load image
  image = plt.imread('image.jpg')

  width = image.shape[1]
  height = image.shape[0]

  peg_width = root_peg_width * max(width, height)

  for brick_line in data['brick_lines']:
    for i, brick in enumerate(brick_line):
      # choose a random color from the palette
      color = data['palette'][np.random.randint(len(data['palette']))]

      # peg looks like [[[x1,y1],[x2,y2]]]
      # it's normalized to be between 0 and 1 
      # so we need to multiply by width and height
      x1 = int(brick[0][0] * width)
      y1 = int(brick[0][1] * height)
      x2 = int(brick[1][0] * width)
      y2 = int(brick[1][1] * height)

      vector_direction = np.array([x2-x1, y2-y1])
      inverse_vector_direction = np.array([y2-y1, x1-x2])

      # normalize vector direction
      vector_direction = vector_direction / np.linalg.norm(vector_direction)
      inverse_vector_direction = inverse_vector_direction / np.linalg.norm(inverse_vector_direction)

      # we have two inverse directions, one for front, one for back
      front_inverse_vector_direction = inverse_vector_direction
      back_inverse_vector_direction = inverse_vector_direction

      prev_x1 = -1
      prev_y1 = -1

      # if the first point is the same as the last peg second point
      if i > 0 and brick[0] == brick_line[i-1][1]:
        # then we need to adjust the front inverse vector direction
        prev_x1 = int(brick_line[i-1][0][0] * width)
        prev_y1 = int(brick_line[i-1][0][1] * height)
      elif i == 0 and brick[0] == brick_line[-1][1]:
        # then we need to adjust the front inverse vector direction
        prev_x1 = int(brick_line[len(brick_line) - 1][0][0] * width)
        prev_y1 = int(brick_line[len(brick_line) - 1][0][1] * height)

      if prev_x1 != -1 and prev_y1 != -1:
        total_vector = np.array([x2-prev_x1, y2-prev_y1])

        inverse_total_vector = np.array([total_vector[1], -total_vector[0]])

        # normalize vector direction
        inverse_total_vector = inverse_total_vector / np.linalg.norm(inverse_total_vector)

        front_inverse_vector_direction = inverse_total_vector

      next_x2 = -1
      next_y2 = -1

      # if the second point is the same as the next peg first point
      if i < len(brick_line) - 1 and brick[1] == brick_line[i+1][0]:
        # then we need to adjust the back inverse vector direction
        next_x2 = int(brick_line[i+1][1][0] * width)
        next_y2 = int(brick_line[i+1][1][1] * height)
      elif i == len(brick_line) - 1 and brick[1] == brick_line[0][0]:
        # then we need to adjust the back inverse vector direction
        next_x2 = int(brick_line[0][1][0] * width)
        next_y2 = int(brick_line[0][1][1] * height)


      if next_x2 != -1 and next_y2 != -1:
        total_vector = np.array([next_x2-x1, next_y2-y1])
        
        inverse_total_vector = np.array([total_vector[1], -total_vector[0]])

        # normalize vector direction
        inverse_total_vector = inverse_total_vector / np.linalg.norm(inverse_total_vector)

        back_inverse_vector_direction = inverse_total_vector


      # we need to draw a filled polygon
      # so we need to find the 4 corners of the polygon
      # and then fill it

      # find the 4 corners of the polygon
      # first corner
      p_x1 = x1 + front_inverse_vector_direction[0] * peg_width
      p_y1 = y1 + front_inverse_vector_direction[1] * peg_width

      # second corner
      p_x2 = x1 - front_inverse_vector_direction[0] * peg_width
      p_y2 = y1 - front_inverse_vector_direction[1] * peg_width

      # third corner
      p_x3 = x2 - back_inverse_vector_direction[0] * peg_width
      p_y3 = y2 - back_inverse_vector_direction[1] * peg_width

      # fourth corner
      p_x4 = x2 + back_inverse_vector_direction[0] * peg_width
      p_y4 = y2 + back_inverse_vector_direction[1] * peg_width

      # draw polygon
      points = np.array([[p_x1,p_y1],[p_x2,p_y2],[p_x3,p_y3],[p_x4,p_y4]], np.int32)

      # draw polygon
      cv2.fillPoly(image, pts =[points], color=color)
      # outline
      cv2.polylines(image, pts =[points], isClosed=True, color=outline_color, thickness=2)

  for brick in data['pegs']:
    # just draw a circle

    # choose a random color from the palette
    color = data['palette'][np.random.randint(len(data['palette']))]

    # fill
    cv2.circle(image, (int(brick[0] * width), int(brick[1] * height)), int(peg_width * 1.5), color, -1) 
    # outline
    cv2.circle(image, (int(brick[0] * width), int(brick[1] * height)), int(peg_width * 1.5), (0,0,0), 2)


  # draw image
  plt.figure(figsize=(10,10))
  plt.imshow(image)
  plt.show()

def quinticBezier(x):
    return 6*x**5 - 15*x**4 + 10*x**3

def input_img(path):
    image = tf.image.decode_png(tf.io.read_file(path))
    image = tf.expand_dims(image, axis=0)
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, [224,224])
    return image

def normalize_image(img):
    grads_norm = img[:,:,0]+ img[:,:,1]+ img[:,:,2]
    grads_norm = (grads_norm - tf.reduce_min(grads_norm))/ (tf.reduce_max(grads_norm)- tf.reduce_min(grads_norm))
    return grads_norm

def convolution(img, kernel):
    # img has shape (x, y), already normalized
    # kernel has shape (3, 3)
    # output has shape (x-2, y-2)
    x = img.shape[0]
    y = img.shape[1]
    output = np.zeros((x-2, y-2))
    for i in range(1, x-1):
        for j in range(1, y-1):
            output[i-1, j-1] = np.sum(img[i-1:i+2, j-1:j+2] * kernel)
    return output

def blur(img):
    # img has shape (x, y), already normalized
    
    # gaussian filter
    kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16
    img = convolution(img, kernel)
    img = convolution(img, kernel)

    # normalize
    img = (img - np.min(img))/ (np.max(img)- np.min(img))

    return img

def get_salient_region(src_img, test_model):
    img = tf.keras.applications.densenet.preprocess_input(src_img)

    result = test_model(img)
    max_idx = tf.argmax(result, axis = 1)

    with tf.GradientTape() as tape:
        tape.watch(img)
        result = test_model(img)
        max_score = result[0, max_idx[0]]
    return tape.gradient(max_score, img)

def turbulence(img):
    # img has shape (x, y), already normalized
    # kernel has shape (5, 5)
    # output has shape (x-2, y-2)
    x = img.shape[0]
    y = img.shape[1]
    output = np.zeros((x-4, y-4))
    for i in range(2, x-2):
        for j in range(2, y-2):
            output[i-2, j-2] = np.max(img[i-2:i+3, j-2:j+3]) - np.min(img[i-2:i+3, j-2:j+3])
    return output

def polarize(img):
    # img has shape (x, y)

    img = turbulence(img)

    # mean
    mean = np.mean(img)
    stddev = np.std(img)

    # mean should be 0.5, stddev should be 0.1
    img = (img - mean) / stddev
    img = 1 / (1 + np.exp(-img))

    img = blur(img)

    img = (img - np.min(img))/ (np.max(img)- np.min(img))

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i, j] = quinticBezier(img[i, j])

    return img

def colorQuant(img, Z, K, criteria):

   ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
   
   unique_colors = np.unique(center, axis=0)

   # Now convert back into uint8, and make original image
   center = np.uint8(center)
   res = center[label.flatten()]
   res2 = res.reshape((img.shape))
   return res2, unique_colors

def plot_images(img1, img2, img3, img4, img5):
   fig, axs = plt.subplots(1, 5, figsize=(15, 15))
   axs[0].imshow(img1)
   axs[1].imshow(img2)
   axs[2].imshow(img3)
   axs[3].imshow(img4)
   axs[4].imshow(img5)
   plt.show()

def get_contours(img):
   img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   ret, im = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY_INV)
   contours, hierarchy  = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
   return contours

def neccessary_score(length):   
   return max(.1, 10.0 / length)

def get_masked_contours(img, saliency):
   # resize saliency to img, in case we ran a filter kernel on it and it changed size
   saliency = cv2.resize(saliency, (img.shape[1], img.shape[0]))

   contours = get_contours(img)
   # we now have a list of contours
   # we want to mask each contour with saliency
   masked = []

   for contour in contours:
      # calculate the mean saliency of the contour
      mean = np.mean(saliency[contour[:, :, 1], contour[:, :, 0]])

      # if the mean is above the threshold, add it to the list
      if mean > neccessary_score(len(contour)):
         masked.append(contour)

   return masked

def draw_contours(src, contours):
   # vibrant pink
   color = (255, 0, 255)
   return cv2.drawContours(src, contours, -1, color, 2)

def resize_to_max(img, max_dim):
   # resize such that the larger dim is max_dim
   if img.shape[0] > img.shape[1]:
      img = cv2.resize(img, (int(img.shape[1] * max_dim / img.shape[0]), max_dim))
   else:
      img = cv2.resize(img, (max_dim, int(img.shape[0] * max_dim / img.shape[1])))
   return img

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def save_pegs(img_src, contours, colors, img_width, 
              img_height, saliency, filename, peg_length, 
              min_brick_line_length, granularity, saliency_cutoff,
              min_dist):
   # resize saliency to img, in case we ran a filter kernel on it and it changed size
   saliency = cv2.resize(saliency, (img_width, img_height))

   # calculates where to place block pegs and saves in json
   pegs = []
   brick_lines = []

   for contour in contours:

      normalized_contour = contour / np.array([img_width, img_height])
      brick_line = []

      # if the segments are too large, we need to add midpoints      
      for i in range(1, len(normalized_contour)):
         point = normalized_contour[i, 0]
         prev = normalized_contour[i-1, 0]
         dist = np.linalg.norm(point - prev)

         if dist > 1.5 * peg_length:
            num_pegs = int(dist / peg_length)
            for j in range(1, num_pegs):
               # insert new points into the contour
               new_point = prev + (point - prev) * (j / num_pegs)
               normalized_contour = np.insert(normalized_contour, i + j - 1, new_point, axis=0)

      start = normalized_contour[0, 0]

      for i in range(1, len(normalized_contour)):
         point = normalized_contour[i, 0]
         dist = np.linalg.norm(point - start)

         if dist > peg_length:
            brick_line.append(np.array([[start[0], start[1]], [point[0], point[1]]]))
            start = point

      if len(brick_line) > min_brick_line_length:
         # if this brick_line is close to a loop, close it
         if np.linalg.norm(start - contour[0, 0]) < peg_length:
            brick_line[-1][1] = brick_line[0][0]

         # lets tighten up the corners, no acute angles
         for i in range(1, len(brick_line) - 1):
            prev = brick_line[i-1]
            curr = brick_line[i]

            a = prev[0] 
            b = curr[0]
            c = curr[1]

            # calculate angle abc
            bc = c - b
            ba = a - b

            angle = np.arccos(np.dot(bc, ba) / (np.linalg.norm(bc) * np.linalg.norm(ba)))

            # if angle is acute
            if angle < np.pi / 2:
               # if the angle is acute, we want to move it such that the angle is 90 degrees
               total_vector = c - a
               half_vector = total_vector / 2
               inverse_half_vector = np.array([-half_vector[1], half_vector[0]])

               option1 = c - half_vector + inverse_half_vector
               option2 = c - half_vector - inverse_half_vector

               if np.linalg.norm(option1 - b) < np.linalg.norm(option2 - b):
                  brick_line[i][0] = option1
                  brick_line[i-1][1] = option1
               else:
                  brick_line[i][0] = option2          
                  brick_line[i-1][1] = option2

         for i in range(len(brick_line)):
            if i < len(brick_line):
               # if the length of this brick is too short, just remove it
               if np.linalg.norm(brick_line[i][0] - brick_line[i][1]) < peg_length / 2 or np.linalg.norm(brick_line[i][0] - brick_line[i][1]) > peg_length * 2:
                  brick_line.pop(i)
                  i -= 1

         brick_lines.append(brick_line)

   # now add circular pegs to non salient space
   all_points = []
   for x in range(granularity):
      for y in range(granularity):
         if y % 2 == 0:
            all_points.append([x / granularity, y / granularity])
         else:
            all_points.append([(x + 0.5) / granularity, y / granularity])

   # shuffle the points
   np.random.shuffle(all_points)

   for point in all_points:
      # check if point is in saliency
      if saliency[int(point[1] * img_height), int(point[0] * img_width)] < saliency_cutoff:
         # check if point is close to any brick_line
         too_close = False
         for brick_line in brick_lines:
            for line in brick_line:
               # calculate distance from point to line
               a = line[0]
               b = line[1]

               if np.linalg.norm(np.array(a) - point) < min_dist:
                  too_close = True
               if np.linalg.norm(np.array(b) - point) < min_dist:
                  too_close = True

            if too_close:
               break
         # check if point is close to any other pegs
         for peg in pegs:
            if np.linalg.norm(np.array(peg) - point) < min_dist:
               too_close = True
               break

         if not too_close:
            pegs.append([point[0], point[1]])

   # save to json
   result = {
      'filename': img_src,
      'brick_lines': brick_lines,
      'pegs': pegs,
      'palette': np.array(colors)
   }

   with open(filename, 'w') as outfile:
      json.dump(result, outfile, cls=NumpyEncoder)

def process(image, test_model, save_filename, 
            peg_length=0.08, num_k=3, alpha=1.5,
            beta=0, min_brick_line_length=5,
            granularity=10, saliency_cutoff=0.5,
            min_dist=0.05):
   filename = 'image.jpg'
   urllib.request.urlretrieve(image, filename)

   img_tf = input_img(filename)
   grads = get_salient_region(img_tf, test_model)
   saliency = polarize(blur(normalize_image(grads[0])))

   img = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB) 

   # resize such that the larger dim is 200
   img = resize_to_max(img, 300)

   # add guassian blur
   blurred = cv2.GaussianBlur(img, (5, 5), 0)
   blurred = cv2.GaussianBlur(blurred, (5, 5), 0)

   meanshifted, colors = colorQuant(blurred, np.float32(blurred.reshape((-1,3))), num_k, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0))

   # increase contrast of meanshifted
   meanshifted = cv2.convertScaleAbs(meanshifted, alpha=alpha, beta=beta)   

   src_img = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
   src_img = resize_to_max(src_img, 300)
   contours_src = draw_contours(src_img, get_contours(meanshifted))
   contours_saliency = draw_contours(np.zeros(img.shape, dtype=np.uint8), get_contours(meanshifted,))
   contours_saliency_masked = draw_contours(np.zeros(img.shape, dtype=np.uint8), get_masked_contours(meanshifted, saliency))

   plot_images(img, meanshifted, contours_src, contours_saliency, contours_saliency_masked)

   save_pegs(image, get_masked_contours(meanshifted, saliency), colors, meanshifted.shape[1],
              meanshifted.shape[0], saliency, save_filename, peg_length, 
              min_brick_line_length, granularity, saliency_cutoff,
              min_dist)

