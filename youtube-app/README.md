![Docker Stars Shield](https://img.shields.io/docker/stars/recodeuk/youtube-dl-flask.svg?style=flat-square)
![Docker Pulls Shield](https://img.shields.io/docker/pulls/recodeuk/youtube-dl-flask.svg?style=flat-square)


# TL;DR Why choose this 
This fork also builds upon the current infrastructure to provide an experience where the user can search youtube and view the thumbnail & title and choose to download a video.


# youtube-downloader-flask

Very spartan Web and REST interface for downloading youtube videos onto a server. [`flask`](https://github.com/pallets/flask) + [`youtube-dl`](https://github.com/rg3/youtube-dl).

![main page](youtube-dl-flask.png)

## Running

### Docker CLI

This example uses the docker run command to create the container to run the app. Here we also use host networking for simplicity. Also note the `-v` argument. This directory will be used to output the resulting videos

```shell
docker run -d --net="host" --name youtube-dl -v /home/core/youtube-dl:/youtube-dl recodeuk/youtube-dl-flask
```

### Docker Compose

This is an example service definition that could be put in `docker-compose.yml`. This service uses a VPN client container for its networking.

```yml
  youtube-dl:
    image: "recodeuk/youtube-dl-flask"
    volumes:
      - /home/core/youtube-dl:/youtube-dl
    restart: always
```



## Usage

### Start a download remotely

Downloads can be triggered by supplying the `{{url}}` of the requested video through the Web UI or through the REST interface via curl, etc.

#### HTML

Just navigate to `http://{{host}}:8080/youtube-dl` and enter the requested `{{url}}`.

#### Curl

```shell
curl -X POST --data-urlencode "url={{url}}" http://{{host}}:8080/youtube-dl/q
```

#### Fetch

```javascript
fetch(`http://${host}:8080/youtube-dl/q`, {
  method: "POST",
  body: new URLSearchParams({
    url: url,
    format: "bestvideo"
  }),
});
```

#### Bookmarklet

Add the following bookmarklet to your bookmark bar so you can conviently send the current page url to your youtube-dl-server instance.

```javascript
javascript:!function(){fetch("http://${host}:8080/youtube-dl/q",{body:new URLSearchParams({url:window.location.href,format:"bestvideo"}),method:"POST"})}();
```

## Implementation

The server uses [`flask`](https://github.com/pallets/flask) for the web framework and [`youtube-dl`](https://github.com/rg3/youtube-dl) to handle the downloading. The integration with youtube-dl makes use of their [python api](https://github.com/rg3/youtube-dl#embedding-youtube-dl).

This docker image is based on [`python:alpine`](https://registry.hub.docker.com/_/python/) and consequently [`alpine:3.8`](https://hub.docker.com/_/alpine/).

[1]:youtube-dl-server.png
