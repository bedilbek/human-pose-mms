import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import requests
import typer


def resize_short_within(img, short_size=416, max_size=1024, stride=1):
    h, w, _ = img.shape
    im_size_min, im_size_max = (h, w) if w > h else (w, h)
    scale = float(short_size) / float(im_size_min)
    if np.round(scale * im_size_max / stride) * stride > max_size:
        # fit in max_size
        scale = float(np.floor(max_size / stride) * stride) / float(im_size_max)
    new_w, new_h = (int(np.round(w * scale / stride) * stride),
                    int(np.round(h * scale / stride) * stride))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def draw_keypoints(img, coords, confidences, keypoint_thresh=0.2):
    joint_visible = confidences[:, :, 0] > keypoint_thresh
    joint_pairs = [[0, 1], [1, 3], [0, 2], [2, 4],
                   [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
                   [5, 11], [6, 12], [11, 12],
                   [11, 13], [12, 14], [13, 15], [14, 16]]

    colors = np.random.randint(0, 255, size=(len(joint_pairs), 3))
    for i in range(coords.shape[0]):
        pts = coords[i]
        for clr, jp in zip(colors, joint_pairs):
            if joint_visible[i, jp[0]] and joint_visible[i, jp[1]]:
                pt1 = tuple(map(int, pts[jp[0]]))
                pt2 = tuple(map(int, pts[jp[1]]))
                cv2.line(img, pt1, pt2, tuple(clr.tolist()), thickness=2)

    return img


def main(
        image_path: Optional[Path] = typer.Option(default='example.jpg', exists=True, file_okay=True, readable=True),
        service_url: Optional[str] = typer.Option(default='http://localhost:8080/predictions/posenet/')
):
    file_bytes = image_path.read_bytes()
    response = requests.post(service_url, files={"body": file_bytes})
    if response.status_code == 200:
        data = json.loads(response.content)
        print(data)
        img_array = np.frombuffer(file_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        transformed_img = resize_short_within(img, short_size=512)

        coords = np.array(data['estimations'][0]['coords']).reshape((-1, 17, 2))
        confidences = np.array(data['estimations'][0]['confidences']).reshape((-1, 17, 1))
        ready_img = draw_keypoints(transformed_img, coords, confidences, 0.05)
        cv2.imshow('HUMAN_POSE_JOINTS', ready_img)
        cv2.imwrite('example-out.jpg', ready_img)
        cv2.waitKey()

    else:
        print(f'MMS server response error, status_code:{response.status_code}')
        exit(-1)


if __name__ == '__main__':
    typer.run(main)
