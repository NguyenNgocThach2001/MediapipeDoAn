from PIL import Image

num_key_frames = 10

with Image.open('lightning.gif') as im:
    for i in range(num_key_frames):
        im.seek(im.n_frames // num_key_frames * i)
        im.save('{}.png'.format(i))

