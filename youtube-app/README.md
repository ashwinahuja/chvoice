# Youtube Denoiser and Downloader


## Usage

Running is simple. Clone the reposiory, then the following will spin up a Flask server on your machine (minimal changes needed to make it web-facing).

```shell
python3 app.py
```


## Implementation

The server uses [`flask`](https://github.com/pallets/flask) for the web framework and [`youtube-dl`](https://github.com/rg3/youtube-dl) to handle the downloading. The integration with youtube-dl makes use of their [python api](https://github.com/rg3/youtube-dl#embedding-youtube-dl).

The Flask server allows the user to input a youtube video link (or any other [`supported video website`](http://ytdl-org.github.io/youtube-dl/supportedsites.html)). It then adds this video to a download queue; which downloads the video onto the server, then processes each second of audio with our denoising model, before joining the original video back to the processed audio and delivering a downloadable link to the user.
