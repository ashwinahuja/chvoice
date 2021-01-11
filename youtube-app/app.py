from __future__ import unicode_literals
import os
from queue import Queue
from flask import Flask, request, render_template, send_file
from threading import Thread
import youtube_dl
from collections import ChainMap
import numpy as np
import neural
import subprocess


app = Flask(__name__, static_url_path='/')
audio_process = neural.Neural('static/model-weights.pkl')

status = {}
url2id = {}

app_defaults = {
    'YDL_FORMAT': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]',
    'YDL_EXTRACT_AUDIO_FORMAT': 'wav',
    'YDL_EXTRACT_AUDIO_QUALITY': '94',
    'YDL_RECODE_VIDEO_FORMAT': None,
    'YDL_OUTPUT_TEMPLATE': 'vidx/%(id)s.%(ext)s',
    'YDL_ARCHIVE_FILE': None,
    'YDL_SERVER_HOST': '0.0.0.0',
    'YDL_SERVER_PORT': 8080,
}


@app.route('/response')
def get_status():
    url = request.args.get('url').strip('\"')
    if url in status:
        return render_template('response.html', response=status[url])


@app.route('/')
def dl_queue_list():
    return render_template('index.html')


@app.route('/viddone/<path:filename>', methods=['GET', 'POST'])
def server_viddone(filename):
    return send_file('viddone/'+filename)

@app.route('/dlprocess')
def dlprocess():
    url = request.args.get("url").strip('\"')
    options = {
        'format': request.args.get("format").strip('\"')
    }
    if not url:
        return render_template('added.html', status="Failed", info="Due to : no URL")
    dl_q.put((url, options))
    status[url] = f"Queued for download ({dl_q.qsize()-1} videos ahead of you)"
    while True:
        s = f'{np.random.randint(99999999)}'
        if s not in url2id.values():
            url2id[url] = s
            break

    return ('', 204)


def proc_worker():
    while not done:
        url, options = proc_q.get()
        status[url] = 'Processing audio...'
        audio_process.proc(f'vidx/{url2id[url]}.wav')
        subprocess.call(f'ffmpeg -i ./vidx/{url2id[url]}.mp4 '
                       f'-i ./vidx/{url2id[url]}.wav -c:v copy -c:a aac -map 0:v:0'
                       f' -map 1:a:0 ./viddone/{url2id[url]}.mp4', shell=True)
        subprocess.call(f'rm -r ./vidx/{url2id[url]}*', shell=True)
        status[url] = f'{url2id[url]}.mp4'
        proc_q.task_done()


def dl_worker():
    while not done:
        url, options = dl_q.get()
        download(url, options)
        dl_q.task_done()
        proc_q.put((url, options))


def get_ydl_options(request_options, url):


    request_vars = {
        'YDL_EXTRACT_AUDIO_FORMAT': 'wav',
        'YDL_RECODE_VIDEO_FORMAT': None,
    }

    requested_format = request_options.get('format', 'bestvideo')

    if requested_format in ['aac', 'flac', 'mp3', 'm4a', 'opus', 'vorbis', 'wav']:
        request_vars['YDL_EXTRACT_AUDIO_FORMAT'] = requested_format
    elif requested_format == 'bestaudio':
        request_vars['YDL_EXTRACT_AUDIO_FORMAT'] = 'best'
    elif requested_format in ['mp4', 'flv', 'webm', 'ogg', 'mkv', 'avi']:
        request_vars['YDL_RECODE_VIDEO_FORMAT'] = requested_format

    ydl_vars = ChainMap(request_vars, os.environ, app_defaults)

    postprocessors = []

    if(ydl_vars['YDL_EXTRACT_AUDIO_FORMAT']):
        postprocessors.append({
            'key': 'FFmpegExtractAudio',
            'preferredcodec': ydl_vars['YDL_EXTRACT_AUDIO_FORMAT'],
            'preferredquality': ydl_vars['YDL_EXTRACT_AUDIO_QUALITY']
        })

    if(ydl_vars['YDL_RECODE_VIDEO_FORMAT']):
        postprocessors.append({
            'key': 'FFmpegVideoConvertor',
            'preferedformat': ydl_vars['YDL_RECODE_VIDEO_FORMAT'],
        })

    return {
        'format': ydl_vars['YDL_FORMAT'],
        'keepvideo': True,
        'postprocessors': postprocessors,
        'outtmpl': f'vidx/{url2id[url]}.mp4',
        'download_archive': ydl_vars['YDL_ARCHIVE_FILE'],
        'progress_hooks': [lambda x: update_download_status(x, url)],
    }

def update_download_status(x, url):
    if x['status'] == 'downloading':
        status[url] = f"Downloading {x['_percent_str']} ({x['_speed_str']})"
    elif x['status'] == 'finished':
        status[url] = f"Download finished... queued for processing with {proc_q.qsize()} ahead of you."
    else:
        status[url] = "Something went wrong. Please refresh and try again."


def download(url, request_options):
    opt = get_ydl_options(request_options, url)
    with youtube_dl.YoutubeDL(opt) as ydl:
        ydl.download([url])


dl_q = Queue()
proc_q = Queue()
done = False
dl_thread = Thread(target=dl_worker)
dl_thread.start()

proc_thread = Thread(target=proc_worker)
proc_thread.start()

app_vars = ChainMap(os.environ, app_defaults)

app.run(host=app_vars['YDL_SERVER_HOST'], port=app_vars['YDL_SERVER_PORT'], debug=True)
done = True
dl_thread.join()
