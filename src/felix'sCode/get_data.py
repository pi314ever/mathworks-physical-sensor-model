import matplotlib.pyplot as plt
from paralleldomain.decoding.helper import decode_dataset

dataset = decode_dataset(dataset_path="/Users/felixmeng/Desktop/CME291", dataset_format="dgp")

# load scene
scene = dataset.get_scene(scene_name=dataset.scene_names[0])

# load camera image from frame
frame = scene.get_frame(frame_id=scene.frame_ids[0])
image_data = frame.get_camera(camera_name=frame.camera_names[0]).image

# print camera image
plt.imshow(image_data.rgba)
plt.title(frame.camera_names[0])
plt.show()