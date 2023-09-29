import numpy as np
import cv2
import os
import torch
import matplotlib.pyplot as plt


def make_dir(path):
    # Creates a directory if it doesn't exist already
    if not os.path.exists(path):
        os.makedirs(path)


def cartToPol(x, y):  
    # Convert cartesian to polar coordinates (for optical flow)
    ang = np.arctan2(y, x)
    mag = np.hypot(x, y)
    return mag, ang


def uv_2_rgb(image_uv):
    # Convert the optical flow field into HSV Polar coordinate representation

    uv_shape = image_uv.shape
    hsv = np.zeros((uv_shape[0], uv_shape[1], 3))
    hsv[..., 1] = 255

    mag, ang = cartToPol(image_uv[..., 0], image_uv[..., 1])
    hsv[..., 0] = (ang+np.pi) * 180 / ( 2 * np.pi )
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    hsv = np.round(hsv).astype(np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return bgr


def images_2_video(image_folder, video_name, fps=30):
    """
    Takes the images of a folder and form them into a video.

    :param image_folder: the folder where the images are stores as a string of the path
    :param video_name: the name of the video as a string
    :param fps: the number of fps as int
    """

    image_names = os.listdir(image_folder)
    image_names.sort()
    frame = cv2.imread(os.path.join(image_folder, image_names[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, 0, fps, (width,height))

    for image in image_names:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()


def all_images_2_video(dataset_path, subdir, res, video_folder, fps):
    """
    Creates videos of all subdirectories of the dataset_path.

    :param dataset_path: the path to the (DAVIS) dataset as a string
    :param subdir: the name of the subdirectory of DAVIS as a string. whether 'train', 'val' or 'test'
    :param res: the resolution of the images as string, e.g. '176p'
    :param video_folder: the name of the folder where the videos should be stored as a string
    :param fps: the number of fps as int
    """

    image_path = os.path.join(dataset_path, subdir, res)
    video_path = os.path.join(dataset_path, subdir, video_folder, res)

    make_dir(video_path)

    print('Image path: {}'.format(image_path))
    print('Video path: {}'.format(video_path))

    subdirs = os.listdir(image_path)
    print('Number of subdirectories: {}'.format(len(subdirs)))

    for subdir in tqdm(subdirs):
        if subdir == '.DS_Store': continue
        image_folder = os.path.join(image_path, subdir)
        video_name = os.path.join(video_path, subdir + '.avi')
        images_2_video(image_folder, video_name, fps)


def checkpoint(folder, model, filename):
    # Creates a checkpoint of the model
    filename = os.path.join(folder, filename)
    torch.save(model.state_dict(), filename)


def resume(model, folder, filename):
    # Loads a current (already trained) model.
    filename = os.path.join(folder, filename)
    if not torch.cuda.is_available():
        model.load_state_dict(torch.load(filename, map_location = torch.device('cpu')))
    else:
        model.load_state_dict(torch.load(filename))


def load_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def save_image(image, image_path):
    cv2.imwrite(image_path, image)


def plot_loss(architecture, train_loss, val_loss):
    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.title(f'{architecture} loss - MSE')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(('Train loss', 'Validation loss'))
    plt.yscale('log')
    plt.grid(True)
    plt.show()