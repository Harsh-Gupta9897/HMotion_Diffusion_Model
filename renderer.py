import numpy as np
import os
import cv2
from PIL import Image
import json
import imageio


# This function will plot the landmarks as a stick man for visualizations 
def plot_landmarks(img, landmarks, idx): 
    def make_coord(data_point):
        x = int(data_point[0])
        y = int(data_point[1])
        return (x,y)


    stick_color_r = (255,0,0)
    stick_color_g1 = (0,255,0)
    stick_color_g2 = (0,180,0)
    stick_color_b1 = (0,0,255)
    stick_color_b2 = (0,0,180)

    # First drawing lines 
    cv2.line(img,make_coord(landmarks['nose'][idx]),make_coord(landmarks['neck_mid'][idx]),stick_color_r,5) # Neck line
    cv2.line(img,make_coord(landmarks['left_shoulder'][idx]),make_coord(landmarks['right_shoulder'][idx]),stick_color_r,5) # Shoulder 
    cv2.line(img,make_coord(landmarks['left_shoulder'][idx]),make_coord(landmarks['left_hip'][idx]),stick_color_r,5) # Body
    cv2.line(img,make_coord(landmarks['right_shoulder'][idx]),make_coord(landmarks['right_hip'][idx]),stick_color_r,5) # Body
    cv2.line(img,make_coord(landmarks['left_hip'][idx]),make_coord(landmarks['right_hip'][idx]),stick_color_r,5) # Hip
    
    # Drawing skeleton of lines 
    cv2.line(img,make_coord(landmarks['left_shoulder'][idx]),make_coord(landmarks['left_elbow'][idx]),stick_color_g1,5) # left upper hand
    cv2.line(img,make_coord(landmarks['right_shoulder'][idx]),make_coord(landmarks['right_elbow'][idx]),stick_color_g1,5) # right upper hand
    cv2.line(img,make_coord(landmarks['left_elbow'][idx]),make_coord(landmarks['left_hand'][idx]),stick_color_g2,5) # left hand
    cv2.line(img,make_coord(landmarks['right_elbow'][idx]),make_coord(landmarks['right_hand'][idx]),stick_color_g2,5) # right hand  
    cv2.line(img,make_coord(landmarks['left_hip'][idx]),make_coord(landmarks['left_knee'][idx]),stick_color_b1,5) # left hand
    cv2.line(img,make_coord(landmarks['right_hip'][idx]),make_coord(landmarks['right_knee'][idx]),stick_color_b1,5) # right hand 
    cv2.line(img,make_coord(landmarks['left_knee'][idx]),make_coord(landmarks['left_foot_mean'][idx]),stick_color_b2,5) # left hand
    cv2.line(img,make_coord(landmarks['right_knee'][idx]),make_coord(landmarks['right_foot_mean'][idx]),stick_color_b2,5) # right hand  


# this function will render the video sequence given the set of landmark points 
def render_seq(landmark_seq, gif_save_path):
    img_size = 512
    img_renders = []
    n_frames = len(landmark_seq['right_shoulder'])

    print("len landmarks: {}".format(len(landmark_seq['right_knee'])))
    for idx in range(0, n_frames):
        img = np.ones((img_size, img_size, 3)) * 192 
        plot_landmarks(img, landmark_seq, idx)
        img = img.astype(np.uint8)
        img_renders.append(Image.fromarray(img))

    print("processed img render len: {}".format(len(img_renders)))
    imageio.mimsave(gif_save_path, img_renders, fps=10)

# This function will load the keypoints for the video given the path for the keypoints 
def load_keypoints(kp_path):
    with open(kp_path, 'r') as fp:
        kps = json.load(fp)
        print("loaded json keys: {}".format(kps.keys()))
        print("loaded json shape: {}".format(len(kps['nose'])))

    return kps

def test_main():
    kp_path = './data/uptown_funk.json' 
    kps = load_keypoints(kp_path)
    save_path = './data/untowm_funk.gif'

    print("saving the rendered keypoints video at: {}".format(save_path))
    print("kps:",  kps)
    render_seq(kps, save_path)


if __name__ == "__main__":
    test_main()
    



