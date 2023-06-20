#
# 1st : make json files from movie file runing openpose
#   openpose-gpu\bin\OpenPoseDemo.exe -model_folder openpose-gpu\models --video input.mp4 -write_json output_dir -part_candidates true
#
# 2nd : run this script
#   change : file_json   as input json directory made by 1st
#            file_movie  as input movie file
#            write_movie as output movie file
#            sx0,sy0 sx1,sy1 sx2,sy2 sx3,sy3 as source coordinates to convert affine
#

import json
import cv2
import numpy as np


file_json   = '20230507_6_json/20230507_6_{}_keypoints.json'	# 000000000000
file_movie  = '20230507_6.mp4'
write_movie = '20230507_6_birdview.mp4'
start_frame   = 3			#Start frame
output_frames = 1100			#Number of output frames

sx0 =  600
sy0 =  105
sx1 =  871
sy1 =  102
sx2 = 1500
sy2 =  564
sx3 =   46
sy3 =  563

dx0 =  650
dy0 =  450
dx1 =  750
dy1 =  450
dx2 =  750
dy2 =  700
dx3 =  650
dy3 =  700

orig = np.float32([[sx0, sy0], [sx1, sy1], [sx2, sy2], [sx3,sy3]])
tran = np.float32([[dx0, dy0], [dx1, dy1], [dx2, dy2], [dx3,dy3]])
affine = cv2.getPerspectiveTransform(orig, tran)
print(orig)
print(tran)
print(affine)




def convert_1d_to_2d(l, cols):
  return [l[i:i+cols] for i in range(0, len(l), cols)]


# Input movie
mov = cv2.VideoCapture(file_movie)
fps = mov.get(cv2.CAP_PROP_FPS)
w   = int(mov.get(cv2.CAP_PROP_FRAME_WIDTH))
h   = int(mov.get(cv2.CAP_PROP_FRAME_HEIGHT))
mov.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

# Output movie
fourcc    = cv2.VideoWriter_fourcc(*'X264')
out_movie = cv2.VideoWriter(write_movie, fourcc, fps, (w,h))


# process every frame
frame = 3
while True :
  ret, img = mov.read()
  if not ret :
    while cv2.waitKey(100) == -1 :
      pass
    break

  # read json file
  fname_json = file_json.format(str(frame).zfill(12))
  #print(fname_json)

  in_json = open(fname_json,'r')
  js1     = json.load(in_json)

  pose = js1['people']
  #print(pose)
  num = len(pose)
  #print("num=" + str(num))

  for p in range(0, num):
    pose_key = convert_1d_to_2d(pose[p]['pose_keypoints_2d'],3)

    #print("Left:" + str(pose_key[21]) + "  Right:" + str(pose_key[24]))
    #no_x = pose_key[0][0]   # Nose
    #no_y = pose_key[0][1]
    ne_x = int(pose_key[1][0])   # Neck
    ne_y = int(pose_key[1][1])
    l_x  = int(pose_key[21][0])  # Left heel
    l_y  = int(pose_key[21][1])
    r_x  = int(pose_key[24][0])  # Right heel
    r_y  = int(pose_key[24][1])

    if (l_x == 0) and (l_y == 0) :
      c_x = r_x
      c_y = r_y
    elif (r_x == 0) and (r_y == 0) :
      c_x = l_x
      c_y = l_y
    else :
      c_x = int((l_x + r_x) / 2.0)
      c_y = int((l_y + r_y) / 2.0)

    if (c_x == 0) and (c_y == 0):
      continue


    # Uniform color
    dst_color = (255,0,0)
    #if ne_x != 0 and ne_y != 0 :
    #    dst_color = img[ne_y+3][ne_x]
    #    #cv2.rectangle(img, (ne_x-2, ne_y+3-2), (ne_x+2, ne_y+3+2), (0,0,0) ,-1)

    # Mask face
    ##rx = abs(no_x - ne_x)
    ##ry = abs(no_y - ne_y)
    ##r  = max(rx, ry)
    ##print(no_x, no_y, ne_x, no_y, r)
    ##if no_x != 0 and no_y != 0 and ne_x != 0 and ne_y != 0 :
    ##  cv2.rectangle(img, (int(no_x-r), int(no_y-r)), (int(no_x+r), int(no_y+r)), (200,200,200), -1)
    ####r = 15
    ####cv2.rectangle(img, (int(ne_x-r), int(ne_y-r*2)), (int(ne_x+r), int(ne_y)), (200,200,200), -1)

    cv2.rectangle(img, (l_x-5, l_y-5), (l_x+5, l_y+5), (255,255,0) ,-1)
    cv2.rectangle(img, (r_x-5, r_y-5), (r_x+5, r_y+5), (0,0,255)   ,-1)
    cv2.rectangle(img, (c_x-3, c_y-3), (c_x+3, c_y+3), (255,0,0)   ,-1)
    if ne_x != 0 and ne_y != 0 :
      if l_x != 0 and l_y != 0 and r_x != 0 and r_y != 0 :
        cv2.line(img, (l_x, l_y), (r_x, r_y), (0,255,255))
        #dst_color = img[int(ne_y+10)][int(ne_x)]
      if l_x != 0 and l_y != 0 :
        cv2.line(img, (ne_x, ne_y), (l_x, l_y), (0,255,255))
      if r_x != 0 and r_y != 0 :
        cv2.line(img, (ne_x, ne_y), (r_x, r_y), (0,255,255))


    # bird view
    c = np.float32([[[c_x, c_y]]])
    #print(c)
    dst = cv2.perspectiveTransform(c, affine)
    #print("  -> ", dst)
    #cv2.rectangle(img, (int(dst[0][0][0])-3, int(dst[0][0][1])-3), (int(dst[0][0][0])+3, int(dst[0][0][1])+3), (int(dst_color[0]), int(dst_color[1]), int(dst_color[2]))  ,-1)
    cv2.rectangle(img, (int(dst[0][0][0])-3, int(dst[0][0][1])-3), (int(dst[0][0][0])+3, int(dst[0][0][1])+3), dst_color  ,-1)


  # src view frame
  cv2.line(img, (sx0, sy0), (sx1, sy1), (0,255,255))
  cv2.line(img, (sx1, sy1), (sx2, sy2), (0,255,255))
  cv2.line(img, (sx2, sy2), (sx3, sy3), (0,255,255))
  cv2.line(img, (sx3, sy3), (sx0, sy0), (0,255,255))

  # bird view frame
  cv2.rectangle(img, (dx0, dy0), (dx2, dy2), (0,0,0))

  #cv2.imwrite(write_jpeg,img)
  out_movie.write(img)

  cv2.imshow('color', img)

  frame += 1
  if frame > output_frames :
    break

  # quit for 'q'
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

out_movie.release()
mov.release()
cv2.destroyAllWindows()



exit()



# part_candidates
LHeel = js1['part_candidates'][0]['21']
LHeel2 = convert_1d_to_2d(LHeel, 3)
print(LHeel2)

RHeel = js1['part_candidates'][0]['24']
RHeel2 = convert_1d_to_2d(RHeel, 3)
print(RHeel2)


img     = cv2.imread(file_jpeg)


#nlist = len(LHeel2)
#for i in range(0, nlist):
#  print("i:" + str(i) + "  Left:" + str(LHeel2[i][0]) + "," + str(LHeel2[i][1]) + "  Right:" + str(RHeel2[i][0]) + "," + str(RHeel2[i][1]))



for x, y, c in LHeel2:
  print("x = " + str(x) + " : y = " + str(y) + " : c = " + str(c))
  cv2.rectangle(img, (int(x)-3, int(y)-3), (int(x)+3, int(y)+3), (255,255,0),-1)

for x, y, c in RHeel2:
  print("x = " + str(x) + " : y = " + str(y) + " : c = " + str(c))
  cv2.rectangle(img, (int(x)-3, int(y)-3), (int(x)+3, int(y)+3), (0,0,255),-1)



#affine transform
#https://imagingsolution.net/program/python/numpy/solve_affine_matrix/
#https://imagingsolution.net/program/python/numpy/solve_homography_matrix/

#af_src = np.array([
#  [ sx0, sy0,   1,   0,   0,   0, -sx0*dx0, -sy0*dx0],
#  [   0,   0,   0, sx0, sy0,   1, -sx0*dy0, -sy0*dy0],
#  [ sx1, sy1,   1,   0,   0,   0, -sx1*dx1, -sy1*dx1],
#  [   0,   0,   0, sx1, sy1,   1, -sx1*dy1, -sy1*dy1],
#  [ sx2, sy2,   1,   0,   0,   0, -sx2*dx2, -sy2*dx2],
#  [   0,   0,   0, sx2, sy2,   1, -sx2*dy2, -sy2*dy2],
#  [ sx3, sy3,   1,   0,   0,   0, -sx3*dx3, -sy3*dx3],
#  [   0,   0,   0, sx3, sy3,   1, -sx3*dy3, -sy3*dy3],
#])
#af_dst = np.array([dx0, dy1, dx1, dy1, dx2, dy2, dx3, dy3]).T
#ans    = np.matmul(np.linalg.inv(af_src), af_dst)
#affine = np.array([[ans[0], ans[1], ans[2]], [ans[3], ans[4], ans[5]], [ans[6], ans[7], 1]])
#print(affine)


#OpenCV
#https://watlab-blog.com/2019/06/01/projection-transform/

i_trans = cv2.warpPerspective(img, affine, (1280, 800))
cv2.imshow('color', i_trans)
while True:
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break
exit()

# bird view
c = np.array([c_x, c_y, 1]).T
print("c_x,c_y : ", c_x, c_y)
c_af = np.matmul(affine, c)
#print(c_af)
c_af_x = c_af[0]/c_af[2]
c_af_y = c_af[1]/c_af[2]
print("  -> ", c_af_x, c_af_y)


