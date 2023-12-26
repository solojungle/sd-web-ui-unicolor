<div align="center">

![logo](.github/images/logo.png)

unicolor: A stable diffusion extension, it implements luckyhzt's [Unicolor Colorizer](https://github.com/luckyhzt/unicolor).


<h3>

[Features](/features) | [Installation](/install)

</h3>

[![GitHub Repo stars](https://img.shields.io/github/stars/solojungle/sd-web-ui-unicolor)](https://github.com/solojungle/sd-web-ui-unicolor/stargazers)
[![Discord](https://img.shields.io/twitter/follow/aliawari)](https://x.com/follow/aliawari)

</div>

---

I wanted a colorizer that worked easily on an M1 Mac (as well as other systems). I remembered how easy it is to setup a1111, and I did not find any colorizers on the extensions list, so I implemented this one!

There will definitely be bugs so please either make an issue or hit me up on twitter.

## Features

### Stroke Based Coloring (coming soon...)

The original repo supports this, but since Automatic1111 uses Gradio version 3.x. the only way to have a canvas would be to build a custom component that would be out of date as soon as they update to a newer version of Gradio. In Gradio 4.x+ there is a the ImageEditor component which makes things 10x easier. Will be waiting for that update.

### Example Based Coloring

Upload an image close in style to what you want it to look like and the model will create strokes for you resembling it.

### Text Based Coloring

Type descriptions and it will work


## Installation

Follow the guide on how to install extensions [here](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Extensions).

